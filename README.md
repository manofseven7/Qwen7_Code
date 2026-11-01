# Co-Adaptive Privacy-Compression Allocation

Reproducibility code for the IEEE Access manuscript:

> Co-Adaptive Privacy-Compression Allocation for Federated Instruction Tuning
> of Large Language Models via Domain-Sensitive Gradient Masking

This repository contains only the code and recorded outputs needed for the
reported Qwen2.5-7B/MedMCQA validation and the mechanism-matched privacy
accounting check. Model weights and dataset records are not redistributed.

## Repository structure

```text
.
├── configs/                     # frozen experiment metadata
├── docs/                        # protocol and interpretation boundaries
├── experiments/
│   ├── qwen7_standard_gate.py   # LoRA development gate; produces rank-32 adapter
│   ├── qwen7_allocation_matrix.py # joint/rank-only/mask-only allocation table
│   └── qwen7_matched_dp.py      # full-client record-DP development check
├── results/
│   └── qwen7_validation_summary.json
├── tests/                       # lightweight accountant tests
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Public resources

- Model: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- MedMCQA project: <https://github.com/medmcqa/medmcqa>
- Dataset mirror used by the scripts:
  <https://huggingface.co/datasets/openlifescienceai/medmcqa>

## Environment

Python 3.10 or newer and a CUDA GPU with sufficient memory for a 7B model are
recommended. The reported run used bfloat16 model loading.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If the model is already stored locally, set `QWEN_MODEL` to its directory.
Otherwise the default Hugging Face model identifier is used.

## Reproduce the reported allocation validation

Run commands from the repository root.

```bash
export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
export QWEN_OUT=outputs/qwen7_standard_gate

python -m experiments.qwen7_standard_gate
export QWEN_ADAPTER=outputs/qwen7_standard_gate/adapter_qv_r32_lr5e5
python -m experiments.qwen7_allocation_matrix
```

The first command creates the frozen rank-32 LoRA adapter selected by the
development protocol. The second command constructs the rank-only, mask-only,
and joint dense-`DeltaW` variants and evaluates all variants on the same
512-item labelled holdout subset. Outputs are written below `QWEN_OUT`.

The expected aggregate results are recorded in
`results/qwen7_validation_summary.json`. Small numerical differences can occur
across PyTorch/CUDA versions because randomized SVD is used.

## Mechanism-matched DP development check

```bash
python -m experiments.qwen7_matched_dp \
  --model "$QWEN_MODEL" \
  --out outputs/qwen7_matched_dp \
  --seed 4101 --train-n 2048 --epochs 3 --rank 32 \
  --lr 5e-4 --clip 1.0 --epsilon 2.0 --delta 1e-5
```

This script uses full-client per-record clipping, one Gaussian release per
epoch, replace-one sensitivity `2C/n`, and non-subsampled Gaussian RDP. It is a
development check of the implemented accounting path. It is not the source of
the manuscript's allocation-accuracy table.

## Interpretation boundary

The reported allocation table does not inject DP noise. It validates public
mask/rank allocation, dense-update reconstruction, payload accounting, and
deterministic evaluation. The formal DP theorem is separate. Do not describe
the table as an end-to-end private federated experiment.

## Data and output policy

The scripts download public MedMCQA records at runtime and write manifests,
predictions, adapter weights, and summaries under `outputs/`. The `outputs/`
directory and model checkpoints are ignored by Git.

## Citation and DOI

Before creating the Zenodo release, replace
`10.5281/zenodo.REPLACE_WITH_FINAL_RECORD` in `CITATION.cff` with the assigned
record DOI. Then use the same DOI in the manuscript's Data and Code
Availability section.

