# L4 VM run — setup & transfer

Everything the Google-Cloud **L4** VM needs to run `vm_run.sh`. The VM does only
prune / fine-tune / dropout-training; all base models are already trained locally, so we
**copy checkpoints** (fastest) rather than retrain.

## 0. What runs (vm_run.sh, in order)
1. **Prune convsep_heavy** (best fold = **f1**): criteria `lin/act/fb` × local+global + `random` local.
1b. **Train linear-attention U-Net** (`networks/linear_attention.py`, linear cross-attention on the skips), folds 0,1,2 (tag `linattn`, trains fresh — no checkpoint needed).
2. **Prune deep & bottleneck**, folds **1,2**: `l1x1/lin/act/fb` × local+global + `random` local (escalation + free-lunch skip).
3. **Train deep & bottleneck with dropout p=0.2**, folds 0,1,2 (tags `mpm_deep_do`, `mpm_bottleneck_do`).

Heavy pruning is **not** here — it runs on the local machine's sweep.

## 1. Code
```bash
git clone https://github.com/thnrigas/morph-unet.git && cd morph-unet
git checkout kasimatis                       # branch with convsep + dropout + convsep-pruning
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision numpy nibabel batchgenerators   # + whatever config.py imports
```

## 2. Data (the big one: ~34 GB)
`config.DATA_DIR = <repo>/data/Task08_HepaticVessel` — gitignored, must be uploaded.
```bash
# from the LOCAL machine (fastest = a GCS bucket in between):
gsutil -m rsync -r data/Task08_HepaticVessel gs://<bucket>/Task08_HepaticVessel
# on the VM:
gsutil -m rsync -r gs://<bucket>/Task08_HepaticVessel data/Task08_HepaticVessel
# (or direct: gcloud compute scp --recurse data/Task08_HepaticVessel <vm>:~/morph-unet/data/ )
```

## 3. Base checkpoints (~235 MB — only the ones the prune steps load)
`results/` is gitignored. Copy exactly these (dropout training in Phase 3 needs none):
```bash
mkdir -p results
gcloud compute scp \
  results/convsep_heavy_f1_best.pth \
  results/mpm_deep_f1_best.pth results/mpm_deep_f2_best.pth \
  results/mpm_bottleneck_f1_best.pth results/mpm_bottleneck_f2_best.pth \
  <vm>:~/morph-unet/results/
```
(Or just `rsync -av results/*_f?_best.pth` — all base checkpoints, ~1.8 GB, harmless.)

## 4. Run
```bash
nohup ./vm_run.sh > vm_run.log 2>&1 &
tail -f vm_run.log
```
Resume-aware: re-running skips finished runs. Missing-checkpoint prune steps are skipped with a
warning (not a crash). Pruned models + `*_prune.json` + `*_scores.json` land in `results/`; the
final `collate_prune.py` prints the summary table.

## 5. Rough L4 timing (24 GB, single GPU, batch 24)
- **Fine-tune per prune run**: 80 ep ≈ 30–45 min (convsep faster, ~20–25 min; bottleneck heaviest).
- **Free-lunch / escalation** cut the count hard — most schemes stop at keep 0.1 or 0.3.
- **Dropout training**: like the originals, ~35 min (convsep-speed) to a few hours (morph, checkpointed) per fold with early stop.
- Phase 1 (convsep, ~6 schemes × escalation): ballpark **3–6 h**.
- Phase 2 (deep+bottleneck × 2 folds × 8 schemes + random): the long pole, **~1–2 days**.
- Phase 3 (6 dropout trainings): **several hours to ~1 day**.

If time is tight, comment out Phase 2/3 tails — Phase 1 (the convsep pruning you asked for first)
is self-contained at the top.

## 6. Pull results back
```bash
gcloud compute scp --recurse <vm>:~/morph-unet/results ./results_vm
```
