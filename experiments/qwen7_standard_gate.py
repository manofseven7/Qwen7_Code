import gc
import hashlib
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = Path(os.environ.get("QWEN_OUT", "outputs/qwen7_standard_gate"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 3101
MAX_LEN = 256
LETTERS = "ABCD"
TRAIN_N, DEV_N, TEST_N = 2048, 256, 512
CONFIGS = [
    {"name": "qv_r16_lr2e5", "rank": 16, "lr": 2e-5},
    {"name": "qv_r16_lr5e5", "rank": 16, "lr": 5e-5},
    {"name": "qv_r32_lr5e5", "rank": 32, "lr": 5e-5},
]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prompt(row):
    return (
        "Answer the following medical multiple-choice question. Reply with only the letter A, B, C, or D.\n"
        f"Question: {row['question']}\nA. {row['opa']}\nB. {row['opb']}\n"
        f"C. {row['opc']}\nD. {row['opd']}\nAnswer:"
    )


def deterministic_subset(ds, n, split_name):
    scored = []
    for i, row in enumerate(ds):
        rid = str(row.get("id", i))
        scored.append((hashlib.sha256((split_name + ":" + rid).encode()).hexdigest(), i))
    scored.sort()
    return [ds[i] for _, i in scored[:min(n, len(scored))]]


def load_roles():
    bundle = load_dataset("openlifescienceai/medmcqa")
    print("AVAILABLE_SPLITS", list(bundle.keys()), {k: len(v) for k, v in bundle.items()}, flush=True)
    train_key = "train"
    dev_key = "validation" if "validation" in bundle else "dev"
    test_key = "test"
    roles = {
        "train": deterministic_subset(bundle[train_key], TRAIN_N, train_key),
        "dev": deterministic_subset(bundle[dev_key], DEV_N, dev_key),
        "sealed_test": deterministic_subset(bundle[test_key], TEST_N, test_key),
    }
    manifest = {
        "protocol": "official splits; label-blind SHA256 subset; test sealed until dev selection",
        "seed": SEED,
        "roles": {k: [str(r.get("id", i)) for i, r in enumerate(v)] for k, v in roles.items()},
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return roles


class TrainSet(Dataset):
    def __init__(self, rows, tok):
        self.items = []
        for row in rows:
            p_ids = tok(prompt(row), add_special_tokens=True, truncation=True, max_length=MAX_LEN-3)["input_ids"]
            a_ids = tok(" " + LETTERS[int(row["cop"])] + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:MAX_LEN]
            labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
            self.items.append((ids, labels))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch, pad):
    m = max(len(x[0]) for x in batch); ids=[]; attn=[]; labels=[]
    for x, y in batch:
        n=m-len(x); ids.append(x+[pad]*n); attn.append([1]*len(x)+[0]*n); labels.append(y+[-100]*n)
    return {"input_ids":torch.tensor(ids), "attention_mask":torch.tensor(attn), "labels":torch.tensor(labels)}


@torch.inference_mode()
def evaluate(model, tok, rows, tag):
    model.eval(); predictions=[]; correct=0
    for start in range(0, len(rows), 8):
        batch=rows[start:start+8]
        enc=tok([prompt(r) for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
        out=model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        for j,row in enumerate(batch):
            raw=tok.decode(out[j,enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
            pred=next((LETTERS.index(c) for c in raw if c in LETTERS), -1); gold=int(row["cop"])
            correct += int(pred==gold); predictions.append({"id":str(row.get("id", start+j)),"gold":gold,"pred":pred,"raw":raw})
    result={"tag":tag,"correct":correct,"n":len(rows),"accuracy":correct/len(rows)}
    (OUT/f"predictions_{tag}.json").write_text(json.dumps(predictions,indent=2))
    print("EVAL",json.dumps(result),flush=True); return result


def fresh_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    )


def train_config(cfg, roles, tok):
    seed_all(SEED); model=fresh_model()
    model=get_peft_model(model,LoraConfig(r=cfg["rank"],lora_alpha=cfg["rank"],lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"]))
    train=TrainSet(roles["train"],tok)
    loader=DataLoader(train,batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(SEED),collate_fn=lambda b:collate(b,tok.pad_token_id))
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg["lr"],weight_decay=0.01)
    opt.zero_grad(set_to_none=True); losses=[]; accum=8; model.train()
    for step,batch in enumerate(loader,1):
        batch={k:v.to("cuda") for k,v in batch.items()}; loss=model(**batch).loss/accum; loss.backward(); losses.append(float(loss.detach())*accum)
        if step%accum==0 or step==len(loader):
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step%128==0: print("TRAIN",cfg["name"],step,len(loader),float(np.mean(losses[-128:])),flush=True)
    dev=evaluate(model,tok,roles["dev"],cfg["name"]+"_dev")
    adapter=OUT/("adapter_"+cfg["name"]); model.save_pretrained(adapter)
    result={"config":cfg,"dev":dev,"mean_loss":float(np.mean(losses)),"last128_loss":float(np.mean(losses[-128:])),"adapter":str(adapter)}
    del model,opt,loader,train; gc.collect(); torch.cuda.empty_cache(); return result


def main():
    seed_all(SEED); roles=load_roles(); tok=AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    base=fresh_model(); base_dev=evaluate(base,tok,roles["dev"],"base_official_dev"); del base; gc.collect(); torch.cuda.empty_cache()
    results=[]
    for cfg in CONFIGS:
        results.append(train_config(cfg,roles,tok)); (OUT/"partial.json").write_text(json.dumps(results,indent=2))
    best=max(results,key=lambda x:(x["dev"]["correct"],-x["config"]["lr"]))
    gain_correct=best["dev"]["correct"]-base_dev["correct"]
    gate_pass=gain_correct>=2
    summary={"protocol_frozen":True,"base_dev":base_dev,"configs":results,"best":best,"gate_rule":"best adapted dev must exceed base by >=2 correct answers","dev_gain_correct":gain_correct,"gate_pass":gate_pass,"sealed_test_opened":False}
    (OUT/"standard_gate_summary.json").write_text(json.dumps(summary,indent=2)); print("STANDARD_GATE_DONE",json.dumps(summary),flush=True)
    # Deliberately do not open the sealed test here. A separate frozen script may do so only after a passing gate.


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (OUT/"FAILED.txt").write_text(traceback.format_exc()); raise
