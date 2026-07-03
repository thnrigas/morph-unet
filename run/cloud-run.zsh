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

# download files from VM
gcloud compute scp athnrigas@deeplearning-4-vm:~/repo/filename ./
gcloud compute scp --recurse athnrigas@deeplearning-4-vm:~/repo/foldername ./
