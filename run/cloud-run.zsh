# google cloud setup
gcloud auth login
gcloud projects list
gcloud config set project project-id
gcloud compute ssh athnrigas@deeplearning-1-vm

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

# morph unet
for cfg in bottleneck balanced heavy; do
    for f in 0 1 2; do
        python3 train_eval.py --tag morphunet_$cfg --morph-unet $cfg --morph-k 3 --fold $f
    done
done

# mean
for t in morphunet_bottleneck morphunet_balanced morphunet_heavy; do
    python3 train_eval.py --fold-mean $t
done

# compare
python3 train_eval.py --compare morphunet_bottleneck_mean_scores.json \
    morphunet_balanced_mean_scores.json morphunet_heavy_mean_scores.json

# download files from google cloud
gcloud compute scp athnrigas@deeplearning-1-vm:~/repo/filename ./
gcloud compute scp --recurse athnrigas@deeplearning-1-vm:~/repo/foldername ./
