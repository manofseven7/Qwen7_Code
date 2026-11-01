# Experimental protocol

## Split construction

Records are ordered by SHA-256 hashes of the public split name and public
record identifier. The Qwen2.5-7B experiment uses 2,048 training records,
256 development records, 512 labelled holdout records, and eight disjoint
public-reference records. Labels do not participate in ordering.

## Evaluation

The prompt requests exactly one answer letter from A to D. Generation is
deterministic. The first valid A-D character in the generated continuation is
extracted; an invalid continuation is counted as incorrect. Accuracy is exact
match against the MedMCQA `cop` field.

## Allocation experiment

LoRA targets query and value projections. The reference checkpoint uses rank
32 and learning rate 5e-5. For each adapted matrix, the allocation stage forms
the dense update `DeltaW = B @ A`, applies a support mask when required, and
re-factorizes the result by randomized SVD. Payload values count bfloat16 LoRA
factor entries and exclude sparse-index metadata.

The allocation table is non-private. Its normalized noise multiplier appears
only in the public allocation surrogate. It is not an empirical DP certificate.

## Privacy development check

`qwen7_matched_dp.py` clips every per-record LoRA-factor gradient to global
norm C, averages all n records, and adds Gaussian noise once per epoch. Under
replace-one adjacency the sensitivity is `2C/n`. No sampling amplification is
used. RDP is accumulated over the predeclared number of epochs and converted
once to `(epsilon, delta)`-DP.

