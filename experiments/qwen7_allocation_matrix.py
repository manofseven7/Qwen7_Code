import gc, hashlib, json, os, time, traceback
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file

from experiments.qwen7_standard_gate import MODEL_PATH, OUT, TrainSet, collate, evaluate, fresh_model
from transformers import AutoTokenizer

MATRIX_OUT = OUT / "allocation_matrix"
MATRIX_OUT.mkdir(parents=True, exist_ok=True)
ADAPTER = Path(os.environ.get("QWEN_ADAPTER", str(OUT / "adapter_qv_r32_lr5e5")))
RHO_GRID = (0.4, 0.6, 0.8)
RANK_GRID = (16, 32)
LAMBDA1, LAMBDA2, A0, Z = 8.0, 0.8, 2.1, 12.28995836
SEED = 3101


def ordered_validation():
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    scored = sorted((hashlib.sha256(("validation:" + str(r.get("id", i))).encode()).hexdigest(), i)
                    for i, r in enumerate(ds))
    held = [ds[i] for _, i in scored[256:768]]
    public = [ds[i] for _, i in scored[768:776]]
    return held, public


def target_name(layer, module):
    return f"model.layers.{layer}.self_attn.{module}.weight"


def adapter_keys(layer, module):
    p = f"base_model.model.model.layers.{layer}.self_attn.{module}"
    return p + ".lora_A.weight", p + ".lora_B.weight"


def support_mask(score, rho):
    flat = score.abs().flatten()
    # Quantile is deterministic and avoids materializing multi-million-element indices.
    threshold = torch.quantile(flat.float(), 1.0 - rho)
    mask = score.abs() >= threshold
    return mask


def public_masks(tok, public_rows):
    torch.manual_seed(SEED)
    model = fresh_model()
    for p in model.parameters(): p.requires_grad_(False)
    named = dict(model.named_parameters())
    targets = []
    for layer in range(28):
        for module in ("q_proj", "v_proj"):
            name = target_name(layer, module)
            named[name].requires_grad_(True); targets.append((layer, module, named[name]))
    ds = TrainSet(public_rows, tok)
    model.train(); model.zero_grad(set_to_none=True)
    for i in range(len(ds)):
        batch = collate([ds[i]], tok.pad_token_id)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        (model(**batch).loss / len(ds)).backward()
    fixed_masks, joint_masks, selections = {}, {}, []
    for layer, module, param in targets:
        h = param.grad.detach().float()
        fixed = support_mask(h, 0.8)
        fixed_masks[(layer,module)] = fixed.cpu()
        best = None
        denom_dim = h.numel()
        for rho in RHO_GRID:
            mask = support_mask(h, rho)
            hm = h * mask
            total = float((hm * hm).sum())
            q = min(32, min(hm.shape)-1)
            _, s, _ = torch.svd_lowrank(hm, q=q, niter=1)
            cumulative = torch.cumsum(s.square(), 0)
            for rank in RANK_GRID:
                kept = float(cumulative[min(rank, len(s))-1]) if total > 0 else 0.0
                erec = 0.0 if total == 0 else max(0.0, 1.0-kept/total)
                m, n = hm.shape
                objective = erec + LAMBDA1 * rank * A0 / (2 * Z * Z) + LAMBDA2 * rank * (m+n) / denom_dim
                cand = (objective, rho, rank, erec, float(mask.float().mean()))
                if best is None or cand < best: best = cand
            del hm, mask
        _, rho, rank, erec, actual_rho = best
        joint_masks[(layer,module)] = support_mask(h, rho).cpu()
        selections.append({"layer":layer,"module":module,"rho":rho,"actual_rho":actual_rho,
                           "rank":rank,"public_erec":erec,"objective":best[0]})
        param.grad = None
        print("ALLOC", selections[-1], flush=True)
    del model, ds, named, targets
    gc.collect(); torch.cuda.empty_cache()
    return fixed_masks, joint_masks, selections


def lowrank_reconstruct(matrix, rank):
    q = min(rank, min(matrix.shape)-1)
    u, s, v = torch.svd_lowrank(matrix.float(), q=q, niter=2)
    return (u * s) @ v.T


def build_dense_variant(state, variant, fixed_masks, joint_masks, selections):
    torch.manual_seed(SEED)
    model = fresh_model()
    named = dict(model.named_parameters())
    selection_map = {(x["layer"],x["module"]):x for x in selections}
    comm_bytes = 0
    with torch.no_grad():
        for layer in range(28):
            for module in ("q_proj", "v_proj"):
                ak, bk = adapter_keys(layer,module)
                a = state[ak].to("cuda", dtype=torch.float32)
                b = state[bk].to("cuda", dtype=torch.float32)
                delta = b @ a  # alpha/r = 32/32 = 1
                if variant == "rank_only":
                    rank, mask = 16, None
                elif variant == "mask_only":
                    rank, mask = 32, fixed_masks[(layer,module)].to("cuda")
                elif variant == "joint":
                    sel = selection_map[(layer,module)]
                    rank, mask = int(sel["rank"]), joint_masks[(layer,module)].to("cuda")
                else:
                    raise ValueError(variant)
                if mask is not None: delta = delta * mask
                recon = lowrank_reconstruct(delta, rank)
                named[target_name(layer,module)].add_(recon.to(named[target_name(layer,module)].dtype))
                m,n = recon.shape; comm_bytes += rank*(m+n)*2
                del a,b,delta,recon,mask
    return model, comm_bytes


def main():
    start=time.time(); tok=AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"
    held, public = ordered_validation()
    fixed_masks, joint_masks, selections = public_masks(tok, public)
    (MATRIX_OUT/"public_allocation.json").write_text(json.dumps(selections,indent=2))
    state=load_file(str(ADAPTER/"adapter_model.safetensors"))
    results=[]
    # Fixed-rank LoRA reference (rank 32, no support mask).
    model=fresh_model(); model=PeftModel.from_pretrained(model,str(ADAPTER),is_trainable=False)
    ref=evaluate(model,tok,held,"matrix_dp_lora_r32")
    ref.update({"method":"DP-LoRA checkpoint reference","rank":32,"mean_rho":1.0,
                "communication_bytes":sum(32*(state[adapter_keys(l,m)[1]].shape[0]+state[adapter_keys(l,m)[0]].shape[1])*2
                                           for l in range(28) for m in ("q_proj","v_proj"))})
    results.append(ref); del model; gc.collect(); torch.cuda.empty_cache()
    for variant in ("rank_only","mask_only","joint"):
        model, comm = build_dense_variant(state,variant,fixed_masks,joint_masks,selections)
        res=evaluate(model,tok,held,"matrix_"+variant)
        res.update({"method":variant,"communication_bytes":comm})
        if variant=="rank_only": res.update({"rank":16,"mean_rho":1.0})
        elif variant=="mask_only": res.update({"rank":32,"mean_rho":0.8})
        else:
            res.update({"mean_rank":float(np.mean([x['rank'] for x in selections])),
                        "mean_rho":float(np.mean([x['rho'] for x in selections]))})
        results.append(res); del model; gc.collect(); torch.cuda.empty_cache()
        (MATRIX_OUT/"partial_results.json").write_text(json.dumps(results,indent=2))
    summary={"scope":"dense-DeltaW allocation/communication validation; formal DP certificate is separate",
             "public_ref_n":len(public),"heldout_n":len(held),"dev_overlap":0,
             "objective":{"lambda1":LAMBDA1,"lambda2":LAMBDA2,"a0":A0,"z":Z},
             "results":results,"elapsed_seconds":time.time()-start}
    (MATRIX_OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    print("MATRIX_DONE",json.dumps(summary),flush=True)


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc(); (MATRIX_OUT/"FAILED.txt").write_text(traceback.format_exc()); raise
