import torch
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr
import numpy as np
from facenet_pytorch import InceptionResnetV1

# Define transform for preprocessing images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load VGGFace model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load sample images
image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")
image3 = Image.open("image3.jpg")
image4 = Image.open("image4.jpg")

# Preprocess images
image1 = transform(image1)
image2 = transform(image2)
image3 = transform(image3)
image4 = transform(image4)

# Add batch dimension
image1 = image1.unsqueeze(0)
image2 = image2.unsqueeze(0)
image3 = image3.unsqueeze(0)
image4 = image4.unsqueeze(0)

# Pass images through the model to get embeddings
embedding1 = model.forward(image1)
embedding2 = model.forward(image2)
embedding3 = model.forward(image3)
embedding4 = model.forward(image4)

# Flatten the embeddings
embedding1 = torch.flatten(embedding1, start_dim=1, end_dim=-1)
embedding2 = torch.flatten(embedding2, start_dim=1, end_dim=-1)
embedding3 = torch.flatten(embedding3, start_dim=1, end_dim=-1)
embedding4 = torch.flatten(embedding4, start_dim=1, end_dim=-1)

# Calculate correlation distance between embeddings to compute RDM
rdm = torch.cdist(embedding1, torch.stack([embedding2, embedding3, embedding4]))
rdm = rdm.flatten()

# Define human behavioral RDM
human_rdm = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1])

# Calculate Spearman rank correlation between network RDM and human behavioral RDM
spearman_corr, _ = spearmanr(rdm, human_rdm)

print(f"Spearman rank correlation between network RDM and human behavioral RDM: {spearman_corr}")

