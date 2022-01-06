## Photovision

More of an artistic and philosophical inquiry than a technical one. Investigation of state-of-the-art ML techniques.
Uses https://github.com/autonomousvision/projected_gan

## Ideas behind the project, motivations, ponderings ##
- I've been a serious hobby photographer for about ten years. I have currently about ~4000 photos in my archive that I consider good. How feasible is it to train a computer to produce more photos that are similar to my photographic work?
- I first had this idea about two years ago. Obviously ML has come light years in that time, and in another two years it will be paradigm-shatteringly effective. 
- Typically training requires enormous datasets, however I'm only human and can only provide  a small amount.
- Some GAN implementations claim to produce quality results even with small datasets; how true is this?
- I mostly have landscape photos in my collmediaection, how will this affect the output?
- Considering other people way smarter than me set up the ML, how much is the output theirs? How much mine?
- How many "familiar" elements will be in the output?
- What feelings will I get when I look at the results? 
- Will I look at a generated image and think, "That's how I would have composed that photographically, yep!"
- Will any of the generated images be indistinguishable from the training data?
- What does the computer's generic understanding of photographic elements that I like to include in my photos (light, framing, bokeh, foreground/background/etc) look like?
- Can I quit photography altogether and just have a machine automate my hobbies?

## Setup ##
First create a VM with GPU (I'm using Google Cloud). This creates a VM with 4 Tesla A100s and 100GB of disk space
```
gcloud compute instances create name --project=photovision-12345 --zone=us-central1-a --machine-type=a2-highgpu-4g --network-interface=network-tier=PREMIUM,subnet=default --maintenance-policy=TERMINATE --service-account=132765626392-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=count=4,type=nvidia-tesla-a100 --create-disk=auto-delete=yes,boot=yes,device-name=name,image=projects/debian-cloud/global/images/debian-10-buster-v20211209,mode=rw,size=100,type=projects/photovision-12345/zones/us-central1-a/diskTypes/pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
```

SSH into the machine, and install necessary libraries:

```
sudo apt-get install git
sudo apt-get install unzip
sudo apt-get install wget
sudo apt-get install bzip2 libxml2-dev
```

Setup Python environment and tools:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh

wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install gdown

git clone https://github.com/austingayler/photovision
conda env create -f environment.yml
conda activate pg
```

Setup GPU driver (this step can be skipped if you're using a specialized VM image):
```
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python install_gpu_driver.py
which nvidia-smi #ensure it's correctly installed
```

Now download your training data and pick a resolution. Resolution will obviously hugely affect training time, so pick smaller resolutions for test runs. Resize the training images, then train away!
```
cd data
gdown https://drive.google.com/file/d/123abc/view?usp=sharing #link to my photos in google drive
unzip landscape.zip
python dataset_tool.py --source=./data/landscape --dest=./data/landscape512.zip --resolution=512x512 --transform=center-crop
(nohup) python train.py --outdir=./training-runs/ --cfg=fastgan_lite --data=./data/landscape512.zip --gpus=1 --batch=8 --mirror=1 --snap=50 --batch-gpu=1 --kimg=2000
```

Now get ready to be weirded out:
```
python gen_images.py --outdir=out --trunc=1.0 --seeds=10-15 --network=./training-runs/00003-fastgan_lite-landscape512-gpus1-batch8/network-snapshot.pkl
```
