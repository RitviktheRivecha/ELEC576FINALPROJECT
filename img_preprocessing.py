import os
import json
import random
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

#PATHS FOR INPUT/OUTPUT, SHOULD MAKE YOUR OWN .env FILE WITH THESE PATHS
INPUT_DIR = os.getenv("INPUT_DIR", "default_path_to_11k_hands_images")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "default_path_to_preprocessed_data")
ANNOTATION_FILE = os.path.join(OUTPUT_DIR, "annotations.json")
print(f"INPUT_DIR: {INPUT_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

#resize
TARGET_SIZE = (256, 256)
NUM_IMAGES_TO_SAMPLE = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

annotations = []

#fixed sample of 1000 images
print("Sampling images...")
all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
sampled_files = random.sample(all_files, NUM_IMAGES_TO_SAMPLE)

#we don't need detailed annotations for this project, so just "A hand" will suffice
print("Processing images...")
for filename in tqdm(sampled_files):
    try:
        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(TARGET_SIZE)
        output_path = os.path.join(PROCESSED_IMAGES_DIR, filename)
        img_resized.save(output_path, format="JPEG")
        annotations.append({
            "image": filename,
            "caption": "A hand."
        })
    except Exception as e:
        print(f"Error processing {filename}: {e}")
print("Saving annotations...")
with open(ANNOTATION_FILE, "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Processing complete. Preprocessed data saved to: {OUTPUT_DIR}")
