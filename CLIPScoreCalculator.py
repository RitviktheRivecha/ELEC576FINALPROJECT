import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def calculate_average_clip_score_fixed(image_folder, text_prompt):
    text = clip.tokenize([text_prompt]).to(device)

    total_score = 0.0

    image_files = sorted([file_name for file_name in os.listdir(image_folder)
                          if file_name.endswith((".png", ".jpg", ".jpeg"))])[:500]

    for file_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, file_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

        total_score += similarity

    average_score = total_score / 500
    return average_score

image_folder = "C:/Users/rchil/Downloads/combined_livingroom_images"
text_prompt = "A living room interior with couches furnished with pillows, a coffee table, lamps, paintings as wall decorations, and large windows letting in sunlight"
average_score = calculate_average_clip_score_fixed(image_folder, text_prompt)

print(f"Average CLIP score for the dataset: {average_score}")
