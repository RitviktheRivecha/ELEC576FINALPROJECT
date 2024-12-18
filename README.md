This is the file repository for our final project for ELEC 576. The contents include:

img_preprocessing.py - Python script for preprocessing datasets

CLIPScoreCalculator.py - Python script for calculating CLIP score given a folder of images and a text caption

FIDScoreCalculator.py - Python script for calculating FID score given a baseline dataset of images and the dataset of images you are calculating the score for. 

livingroom_sample_images.zip - contains 4 folders, each containing a sample of the generated living room images of each of the models 

labeled_hands(2).zip - contains a folder which itself contains a .json file of annotations and a folder of hand images

living_room_dataset.zip - same as above but with living room images

VQGAN.zip
VAE_DECODER_FINAL.zip
SPARSE_ATTENTION_FINAL.zip
COMBINED_ARCH_FINAL.zip
______________________________________

Folders which each contain a .ipynb script to run the model pipeline in colab as well as the output results. All except for VQGAN.zip also contain a .py file with the
modified model architecture. 

TO RUN:

VQGAN.zip - open the .ipynb file in colab. Create a file called "data" in the environment, and then within it create a folder called "images". Import the .json file
from either labeled_hands or living_room_dataset into "data", then open "images" and import the images from the dataset. Run the model pipeline. 

Other models- Import dataset to directory. same as previous model. Run first step of pipeline to clone model files to the colab directory.
Navigate to min-dalle/min_dalle/models/vqgan_detokenizer.py. Replace this file with the file containing the modified architecture, but ensure the name stays the same
for compatibility. Run the pipeline as is from there. 

Resources:

Original Dalle-Mini model by Brett Kuprel used as baseline: https://github.com/kuprel/min-dalle
Hugging Face link with pretrained VAE decoder used in modifications: https://huggingface.co/stabilityai/sd-vae-ft-mse
11k hands dataset: https://sites.google.com/view/11khands
House Rooms image dataset: https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset
