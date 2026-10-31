# Fixed SAM transfer to 16 and 64 bits

The 32-bit seed-60 query result improved with ordinary SAM at rho=0.05. The user
chose to keep seed=60. This next experiment freezes that radius and tests the two
remaining lengths without retraining 32-bit models or preparing another teacher.
This is transfer validation of an existing optimizer, not a new SAM algorithm.

Four students train sequentially: 16-bit ordinary NFD, 16-bit NFD+SAM, 64-bit
ordinary NFD, 64-bit NFD+SAM. Each has 300 epochs, the original RMSprop/schedule,
that length's frozen B1 center/relation/teacher artifacts and original NFD
alpha=0.75, weight=0.01, topk=5, warmup=20, ramp-end=40. Both selections use
validation mAP@ALL, never query. SAM uses two student passes per update; the
controls are matched for epochs and updates, not computational cost.

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
length. Only after these checks pass does nohup launch all four training runs.
Latest pointers are logs/i3_sam_transfer_latest.path and .log_path; the existing
32-bit latest pointers are not changed. Detailed logs live at ROOT/16bit/control.log,
ROOT/16bit/sam.log and their 64bit counterparts. Keep this checkout unchanged
during the experiment. There is no full optimizer/RNG resume support.

Python entry options --dry-run and --smoke-test leave the new output root unused.
--report ROOT checks all completed paired training selections and writes
transfer_report.json/CSV. Training never automatically invokes final query.

After all four training runs complete, --query ROOT separately preflights all
four selected models before iterating any query loader, then evaluates every
control/SAM pair at both lengths. Query is evaluated for all arms regardless of
the validation gain, with fixed rho and original ordinary Hamming ranking.
Outputs are ROOT/query/16bit/control, /16bit/sam and their 64bit counterparts;
logs are ROOT/query/16bit_control.log, etc. An existing ROOT/query is rejected.
--query-report ROOT verifies selection receipts, dataset identities, evaluator
source and saved retrieval-output hashes, and writes query_report.json/CSV.
It reports all four query scores and both SAM-minus-control deltas.

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

Local verification: 37 targeted CPU tests passed (SAM, transfer, NFD and
independent query evaluation). This includes actual tiny paired NFD/SAM training
at both 16 and 64 bits, with teacher-call, initialization and saved-head shape
checks. The Bash starter passed bash -n. Real CIFAR-100 16/64-bit training and
query scoring have not been run on this workstation.
