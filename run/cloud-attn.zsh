#!/usr/bin/env zsh
#
# L4-VM: matched LINEAR-vs-MORPHOLOGICAL attention comparison, from a blank VM.
# ===========================================================================
# Trains 3 arms x 3 folds on the SAME plain-conv (config=none) U-Net backbone, identical
# schedule -> the only difference is the skip gate. Both attention gates start as IDENTITY
# (morph = 2*sigmoid, linear = ReZero gamma=0), so all three share the plain-U-Net baseline.
#
# No checkpoints needed -- everything trains from scratch; the data is downloaded + preprocessed
# on the VM. Run:  zsh run/cloud-attn.zsh   (or copy-paste block by block).
#
set -e

# ---- 1. code: clone + switch to the branch that has the attention models -----------------
git clone "https://github.com/thnrigas/morph-unet.git" repo
cd repo
git checkout kasimatis                        # <-- convsep / linear_attention / identity-init gate
pip install -r requirements.txt --break-system-packages

# ---- 2. data: public MSD bucket -> untar -> nnU-Net-style preprocessing -------------------
mkdir -p data && cd data
curl -O https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar
tar -xf Task08_HepaticVessel.tar
cd ..
python3 run_preprocessing.py                  # writes the preprocessed volumes train_eval reads

# ---- 3. the comparison: 3 arms x 3 folds (CUDA -> keep default --num-workers) -------------
for f in 0 1 2; do
  python3 train_eval.py --tag unet_baseline  --morph-unet none                          --fold $f
  python3 train_eval.py --tag unet_morphattn --morph-unet none --morph-attn --morph-k 3 --fold $f
  python3 train_eval.py --tag unet_linattn   --lin-attn --lin-attn-heads 4              --fold $f
done

# ---- 4. aggregate + honest macro comparison (each training already wrote *_scores.json) ---
for t in unet_baseline unet_morphattn unet_linattn; do
  python3 train_eval.py --fold-mean $t
done
python3 train_eval.py --compare unet_baseline_mean_scores.json \
    unet_morphattn_mean_scores.json unet_linattn_mean_scores.json

# ---- 5. pull results back to your machine (run LOCALLY, not on the VM) --------------------
# gcloud compute scp --recurse athnrigas@deeplearning-4-vm:~/repo/results ./results_vm
