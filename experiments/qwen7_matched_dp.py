"""Matched full-client record-DP LoRA development pilot.

The implementation deliberately does not use sampling amplification.  In each
epoch it clips every record gradient, forms the full-client mean, adds Gaussian
noise once, and performs one AdamW update.  The replace-one sensitivity of the
mean is 2*C/n.  The fixed-cap non-subsampled Gaussian RDP ledger is exported.
"""
import argparse, gc, hashlib, json, math, random, time, traceback
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from coadaptive.accounting import accountant, calibrate

MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"
LETTERS = "ABCD"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prompt(row):
    return ("Answer the following medical multiple-choice question. Reply with only the letter A, B, C, or D.\n"
            f"Question: {row['question']}\nA. {row['opa']}\nB. {row['opb']}\n"
            f"C. {row['opc']}\nD. {row['opd']}\nAnswer:")


def ordered(ds, split):
    scored = sorted((hashlib.sha256((split + ":" + str(r.get("id", i))).encode()).hexdigest(), i)
                    for i, r in enumerate(ds))
    return [ds[i] for _, i in scored]


class Records(Dataset):
    def __init__(self, rows, tok, max_len=256):
        self.items=[]
        for row in rows:
            p=tok(prompt(row), add_special_tokens=True, truncation=True, max_length=max_len-3)["input_ids"]
            a=tok(" "+LETTERS[int(row["cop"])]+tok.eos_token, add_special_tokens=False)["input_ids"]
            ids=(p+a)[:max_len]; labels=[-100]*min(len(p),len(ids))+ids[len(p):]
            self.items.append((ids,labels))
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


def one_record(item, pad):
    x,y=item
    return {"input_ids":torch.tensor([x],device="cuda"),
            "attention_mask":torch.ones((1,len(x)),dtype=torch.long,device="cuda"),
            "labels":torch.tensor([y],device="cuda")}


@torch.inference_mode()
def evaluate(model,tok,rows,out_path):
    model.eval(); preds=[]; correct=0
    for st in range(0,len(rows),8):
        batch=rows[st:st+8]
        enc=tok([prompt(r) for r in batch],return_tensors="pt",padding=True,truncation=True,max_length=256).to("cuda")
        gen=model.generate(**enc,max_new_tokens=3,do_sample=False,pad_token_id=tok.pad_token_id)
        for j,row in enumerate(batch):
            raw=tok.decode(gen[j,enc.input_ids.shape[1]:],skip_special_tokens=True).strip().upper()
            pred=next((LETTERS.index(c) for c in raw if c in LETTERS),-1); gold=int(row["cop"])
            correct+=int(pred==gold); preds.append({"id":str(row.get("id",st+j)),"gold":gold,"pred":pred,"raw":raw})
    out_path.write_text(json.dumps(preds,indent=2))
    return {"correct":correct,"n":len(rows),"accuracy":correct/len(rows)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=MODEL_DEFAULT); ap.add_argument("--out",required=True)
    ap.add_argument("--seed",type=int,default=4101); ap.add_argument("--train-n",type=int,default=2048)
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--rank",type=int,default=32)
    ap.add_argument("--lr",type=float,default=5e-4); ap.add_argument("--clip",type=float,default=1.0)
    ap.add_argument("--epsilon",type=float,default=2.0); ap.add_argument("--delta",type=float,default=1e-5)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed); start=time.time()
    bundle=load_dataset("openlifescienceai/medmcqa")
    tr=ordered(bundle["train"],"train")[:args.train_n]
    val=ordered(bundle["validation"],"validation"); dev=val[:256]; held=val[256:768]
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map={"":0},low_cpu_mem_usage=True)
    model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"]))
    params=[p for p in model.parameters() if p.requires_grad]; data=Records(tr,tok)
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    z=calibrate(args.epochs,args.epsilon,args.delta); sensitivity=2*args.clip/len(data); noise_std=z*sensitivity
    ledger=[]; losses=[]; clip_fractions=[]
    for epoch in range(args.epochs):
        accum=[torch.zeros_like(p,dtype=torch.float32) for p in params]
        epoch_losses=[]; clipped=0
        order=list(range(len(data))); random.Random(args.seed+epoch).shuffle(order)
        model.train()
        for pos,idx in enumerate(order,1):
            model.zero_grad(set_to_none=True)
            loss=model(**one_record(data[idx],tok.pad_token_id)).loss; loss.backward(); epoch_losses.append(float(loss.detach()))
            norm_sq=sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None)
            norm=math.sqrt(norm_sq); scale=min(1.0,args.clip/(norm+1e-12)); clipped+=int(scale<1.0)
            for a,p in zip(accum,params):
                if p.grad is not None: a.add_(p.grad.detach().float(),alpha=scale)
            if pos%128==0: print("DP_PASS",epoch+1,pos,len(data),np.mean(epoch_losses[-128:]),flush=True)
        opt.zero_grad(set_to_none=True)
        gen=torch.Generator(device="cuda").manual_seed(args.seed*1000+epoch)
        for p,a in zip(params,accum):
            mean=a.div_(len(data)); mean.add_(torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=gen),alpha=noise_std)
            p.grad=mean.to(p.dtype)
        opt.step(); model.zero_grad(set_to_none=True)
        best,_=accountant(epoch+1,z,args.delta)
        rec={"epoch":epoch+1,"records":len(data),"clip_C":args.clip,"replace_one_sensitivity":sensitivity,
             "normalized_noise_z":z,"absolute_noise_std_on_mean":noise_std,"epsilon_so_far":best,
             "mean_loss":float(np.mean(epoch_losses)),"clipped_fraction":clipped/len(data)}
        ledger.append(rec); losses.extend(epoch_losses); clip_fractions.append(rec["clipped_fraction"])
        (out/"privacy_ledger.json").write_text(json.dumps(ledger,indent=2)); model.save_pretrained(out/f"adapter_epoch{epoch+1}")
        print("DP_EVENT",json.dumps(rec),flush=True)
    dev_res=evaluate(model,tok,dev,out/"predictions_dev.json")
    held_res=evaluate(model,tok,held,out/"predictions_heldout.json")
    best,orders=accountant(args.epochs,z,args.delta)
    summary={"scope":"matched full-client record-DP LoRA; no sampling amplification", "args":vars(args),
             "adjacency":"replace-one record","noise_events":"once after each full-client clipped mean",
             "train_n":len(data),"dev_n":len(dev),"heldout_n":len(held),"normalized_noise_z":z,
             "sensitivity":sensitivity,"noise_std":noise_std,"accountant_optimum":best,
             "dev":dev_res,"heldout":held_res,"mean_loss":float(np.mean(losses)),
             "mean_clipped_fraction":float(np.mean(clip_fractions)),"elapsed_seconds":time.time()-start}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"accountant_orders.json").write_text(json.dumps(orders,indent=2))
    print("MATCHED_DP_DONE",json.dumps(summary),flush=True)


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc(); raise
