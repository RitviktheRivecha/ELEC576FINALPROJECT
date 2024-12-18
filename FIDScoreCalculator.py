import os
import numpy as np
from scipy.linalg import sqrtm
from torchvision import datasets, transforms, models
import torch
from PIL import Image

def get_activations(image_paths, model, device, batch_size=50):
    activations = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = [(transform(Image.open(path).convert("RGB"))) for path in image_paths]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = model(batch)
            activations.append(features.cpu().numpy())

    return np.concatenate(activations, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_statistics(image_paths, model, device):
    activations = get_activations(image_paths, model, device)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def load_images_from_folder(folder_path):
    return [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith((".png", ".jpg", ".jpeg"))]


if __name__ == "__main__":
    baseline_folder = "C:/Users/rchil/OneDrive/Desktop/living_room_dataset/images"
    generated_folder = "C:/Users/rchil/Downloads/combined_livingroom_images"

    baseline_images = load_images_from_folder(baseline_folder)
    generated_images = load_images_from_folder(generated_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()

    mu1, sigma1 = compute_statistics(baseline_images, inception_model, device)
    mu2, sigma2 = compute_statistics(generated_images, inception_model, device)

    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)
    print(f"FID score: {fid_score}")
