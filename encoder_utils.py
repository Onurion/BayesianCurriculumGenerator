import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torchvision import transforms, datasets
from PIL import Image

# colors = ['black', 'grey', 'green', 'red', 'blue', 'blue']


# Image preprocessing function
def preprocess_image(image):
    # Check if the input is a file path, and if so, read the image
    if isinstance(image, str):
        image = Image.open(image)

    # Check if the input is a NumPy array, and if so, convert it to a PIL image
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # If the image has an alpha channel, remove it
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def extract_features(image, model):
    with torch.no_grad():
        features = model(image)
        features = features.view(features.size(0), -1)
    return features.numpy()

def preprocess_and_extract_features(images, model):
    extracted_features = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        features = extract_features(preprocessed_image, model)
        extracted_features.append(features)
    return extracted_features

def get_updated_latent_output(current_map, encoder, colors, size=(384, 384)):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    cmap = ListedColormap(colors)   
    img = cmap(current_map)
    
    original_image = cv2.resize(img[:,:,:3]*255, size, interpolation=cv2.INTER_NEAREST)
    pil_img = Image.fromarray(cv2.cvtColor(np.uint8(original_image), cv2.COLOR_BGR2RGB).astype('uint8'))
    # predicted_image = np.round(np.transpose(autoencoder(transform(pil_img).unsqueeze(0)).detach().numpy().squeeze(), (1, 2, 0))*255).astype(int)

    with torch.no_grad():
        latent_output = encoder(transform(pil_img).unsqueeze(0)).detach().numpy().squeeze()

    return latent_output

def get_latent_output(img, encoder, size=(224, 224), preprocess = True):
    img = cv2.resize(img, size)

    # Convert image to PyTorch tensor
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))

    # Normalize the tensor
    mean = [0.2007, 0.2061, 0.1817]
    std = [0.2115, 0.2190, 0.1956]

    img_tensor = img_tensor.float() / 255.0
    if preprocess:
        img_tensor = (img_tensor - torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)) / torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)

    # Add batch dimension to the tensor
    img_tensor = img_tensor.unsqueeze(0)

    # Pass the tensor through the VGG network to get the latent output
    with torch.no_grad():
        latent_output = encoder(img_tensor).detach().numpy()

    return latent_output


class DeepAutoencoder2(nn.Module):
    def __init__(self, channels=3):
        super(DeepAutoencoder2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print ("enc: ", encoded.size())
        decoded = self.decoder(encoded)
        # print ("dec: ", decoded.size())
        # stop
        return decoded