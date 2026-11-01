# Fixed SAM transfer to 16 and 64 bits

The 32-bit seed-60 query result improved with ordinary SAM at rho=0.05. The user
chose to keep seed=60. This next experiment freezes that radius and tests the two
remaining lengths without retraining 32-bit models or preparing another teacher.
This is transfer validation of an existing optimizer, not a new SAM algorithm.

The user declined repeating the existing 16/64-bit controls and confirmed the
earlier four-arm command had not been started. The default now trains only two
students sequentially: 16-bit NFD+SAM and 64-bit NFD+SAM. Each has 300 epochs, the original RMSprop/schedule,
that length's frozen B1 center/relation/teacher artifacts and original NFD
alpha=0.75, weight=0.01, topk=5, warmup=20, ramp-end=40. Both selections use
validation mAP@ALL, never query. SAM uses two student passes per update. Historical
NFD controls are reused without retraining or query re-evaluation; the new report
does not claim newly paired initialization, batch order or computation matching.

Only launcher/test/document files change. Core training/evaluation source remains
identical to the completed 32-bit run, preserving its source-check compatibility.

## Historical references and data identity

Default original references are discovered by configuration under
save_nfd_16_64_seed60_ZZdT2Ou0, the original path supplied in this chat. Only
CIFAR-100 b2nfd, seed=60, lengths 16/64, alpha=0.75, weight=0.01 and rho=0
references qualify. Ordinary feature-distillation arms are excluded. An ambiguous
match stops and requires two explicit --nfd-config arguments; no scores select
references. All remaining shared fields are inherited and parent artifacts are
validated before training.

Legacy references without a saved data fingerprint use --dataset-anchor-run:
the completed ordinary 32-bit control, validated using its selection receipt,
model/config/history hashes and training source. Its dataset, split seed, class
count, validation-per-class and split hash must match each target reference.
The expected pixel/label digest comes from that already completed run. Current
raw data is independently fingerprinted and compared; each target's B1 caches
remain independently checked. The anchor neither substitutes 32-bit hash centers
for another length nor proves an otherwise missing historical pixel digest.
It supplies the explicit common dataset identity for this fixed transfer.
All original configs and model files are left unchanged.

## Operations

From the repository root on the GPU machine:

```bash
bash scripts/start_i3_sam_transfer.sh
```

Optional positional arguments are original reference root, GPU number and the
completed 32-bit ordinary control directory. Defaults correspond to the paths
already returned by the user. The starter runs the GPU check and targeted tests,
preflights both lengths, and performs one disposable full-NFD SAM GPU batch per
length. Only after these checks pass does nohup launch the two SAM training runs.
Latest pointers are logs/i3_sam_transfer_latest.path and .log_path; the existing
32-bit latest pointers are not changed. Detailed logs live at ROOT/16bit/sam.log
and ROOT/64bit/sam.log. Keep this checkout unchanged
during the experiment. There is no full optimizer/RNG resume support.

Python entry options --dry-run and --smoke-test leave the new output root unused.
--report ROOT checks both completed SAM training selections and writes
transfer_report.json/CSV. Training never automatically invokes final query.

After both SAM training runs complete, --query ROOT separately preflights the two
selected models before iterating any query loader, then evaluates only those SAM
models regardless of validation gain. Radius and ordinary Hamming ranking stay
fixed. Outputs are ROOT/query/16bit/sam and ROOT/query/64bit/sam; logs are
ROOT/query/16bit_sam.log and ROOT/query/64bit_sam.log. An existing ROOT/query is rejected.
--query-report ROOT verifies selection receipts, dataset identities, evaluator
source and saved retrieval-output hashes, and writes query_report.json/CSV.
It reports both new query scores and SAM-minus-historical-reference deltas.

Historical query references first use saved query_metrics with query_evaluated=true
from each original run's metrics.json or final_query/metrics.json. When those are
absent, the report explicitly uses the six-decimal user-provided query table in
创新点一.docx: 16 bits 0.541192/0.658830/0.640592 and 64 bits
0.598426/0.699022/0.673465 (ALL/100/1000). Source kinds and file hashes where
available are saved in the plan and query report. User-table differences inherit
last-decimal rounding uncertainty. These values never enter model fitting or
epoch selection. No validation score is substituted for a historical query score.

An explicit --paired-control option retains the earlier four-arm mode for
compatibility; the default starter does not pass it. The single-length launcher
similarly accepts --sam-only, which the transfer coordinator always passes in its
default mode. Existing four-arm plans remain readable.

The existing 32-bit query results remain separate and untouched. Any subsequent
method changes informed by these query observations are further development,
not evidence from an untouched blind test. Do not tune radii or pick code lengths
by their query gains. Report single-seed scope and measured training cost.

## Validation

Synthetic tests cover per-length artifact inheritance, fixed rho and seed,
ordinary-FD exclusion, missing/ambiguous reference handling, cross-bit legacy
dataset anchors, mismatch rejection, no training/output on dry runs or smoke
dispatch, no query dispatch during training, all-query preflight ordering,
selected-checkpoint receipts and rejection of modified retrieval artifacts.
Real 16/64-bit CUDA behavior and retrieval efficacy must be checked remotely;
the disposable GPU checks are part of that workflow.

Local verification: 40 targeted CPU tests passed (SAM, transfer, NFD and
independent query evaluation). This includes actual tiny paired NFD/SAM training
at both 16 and 64 bits, with teacher-call, initialization and saved-head shape
checks. The Bash starter passed bash -n. Real CIFAR-100 16/64-bit training and
query scoring have not been run on this workstation.
Additional coverage verifies SAM-only command dispatch, no control directory or
control query creation, historical-result provenance and absence of false pairing
claims in the single-arm report.

## Completed 16/64-bit query transfer (user-reported 2026-11-01)

The user returned the query report for
save/i3_sam_only16_64_seed60_20261031_210100_16097. Only the two SAM models
were newly trained/evaluated. The 16-bit validation selection was epoch 270,
and the 64-bit selection epoch 300. Both query evaluations used 5,000 images
against 45,000 database images, with seed=60 and rho=0.05 unchanged.

| Bits | Method | Query mAP@ALL | Query mAP@100 | Query mAP@1000 |
| --- | --- | ---: | ---: | ---: |
| 16 | Historical B1+NFD | 0.5411921655 | 0.6588297888 | 0.6405917175 |
| 16 | B1+NFD+SAM | 0.5567399196 | 0.6750062540 | 0.6552625632 |
| 64 | Historical B1+NFD | 0.5984255557 | 0.6990219074 | 0.6734654239 |
| 64 | B1+NFD+SAM | 0.6237007732 | 0.7172299271 | 0.6919933940 |

The 16-bit gains were +0.0155477541/+0.0161764652/+0.0146708456;
the 64-bit gains were +0.0252752175/+0.0182080197/+0.0185279701
(ALL/100/1000). Both historical references were actually loaded from their
original metrics.json files, not from the six-decimal document fallback:

- 16-bit original run suffix ac0f8d38af38; metrics SHA256
  d8a5691980dac5bb41b24d0837f98b0299df98af101dd4df1f181173c6ca6fa9.
- 64-bit original run suffix 715253eea766; metrics SHA256
  ff2c5c87f9eccfc311d55582eb07c4112efcec5149d4e84a140ad90c5523f9d8.

The generic report note originally mentioned table-rounding uncertainty even
when both sources were saved metric files. That presentation issue is corrected:
only a real user-table fallback receives that warning. Numerical results,
checkpoints, query codes and scores are unchanged; no re-evaluation is needed.
The local checkout did not independently load the remote artifacts.

Together with the previous 32-bit query result 0.607899/0.706845/0.686009,
all nine fixed-seed query metrics improved over B1+NFD. Preserve this as a
positive CIFAR-100 seed-60 result across the three code lengths. It establishes
the empirical value of adding ordinary SAM in this setting; it does not establish
an original optimizer, multi-seed evidence or a causal explanation of the gain.
The 16/64-bit controls are historical, whereas the 32-bit comparison used a fresh
control reproducing its historical metrics. No additional training is implied
by recording the completed result.
