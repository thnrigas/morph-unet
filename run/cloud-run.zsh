# clone repo
git clone "https://github.com/thnrigas/morph-unet.git" repo
cd repo
pip install -r requirements.txt --break-system-packages

# download data
mkdir data
cd data
curl -O https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar
tar -xvf Task08_HepaticVessel.tar

# preprocessing
cd ..
python3 run_preprocessing.py

# baseline
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag hepatic_base --fold $f
done

# morphological-separable U-Net sweep (3 configs x 5 folds)
# on CUDA leave --num-workers at its default (the 0 was a mac/MPS-only workaround)
for cfg in bottleneck balanced heavy; do
    for f in 0 1 2 3 4; do
        python3 train_eval.py --tag morphunet_$cfg --morph-unet $cfg --morph-k 3 --fold $f
    done
done

# fold means + comparison vs baseline (Dice / ASSD deltas per label)
for t in hepatic_base morphunet_bottleneck morphunet_balanced morphunet_heavy; do
    python3 train_eval.py --fold-mean $t
done
python3 train_eval.py --compare hepatic_base_mean_scores.json \
    morphunet_bottleneck_mean_scores.json morphunet_balanced_mean_scores.json morphunet_heavy_mean_scores.json

gcloud auth login
gcloud projects list
gcloud config set project project-id
gcloud compute ssh athnrigas@deeplearning-4-vm

# download files from VM
gcloud compute scp athnrigas@deeplearning-4-vm:~/repo/filename ./
gcloud compute scp --recurse athnrigas@deeplearning-4-vm:~/repo/foldername ./
