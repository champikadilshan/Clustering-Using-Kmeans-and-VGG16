import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image

# Function to load and preprocess images
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features using VGG16
def extract_features(img_path):
    img_array = load_and_preprocess_image(img_path)
    features = base_model.predict(img_array)
    return features.flatten()

# Path to the folder containing TIFF images
image_folder = 'start_pages'

# Extract features from all images
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith('.tiff')]
features = np.array([extract_features(img_path) for img_path in image_paths])

# Dimensionality reduction using PCA
n_samples, n_features = features.shape
n_components = min(n_samples, n_features, 50)  # Adjust n_components dynamically
pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(features)

# Clustering using KMeans
kmeans = KMeans(n_clusters=5)  # Choose the number of clusters as needed
labels = kmeans.fit_predict(features_reduced)

# Group images based on their cluster labels
grouped_images = {i: [] for i in range(kmeans.n_clusters)}
for img_path, label in zip(image_paths, labels):
    grouped_images[label].append(img_path)

# Create folders and save images in corresponding folders
output_folder = 'clustered_images'
os.makedirs(output_folder, exist_ok=True)

for label, img_paths in grouped_images.items():
    cluster_folder = os.path.join(output_folder, f'cluster_{label}')
    os.makedirs(cluster_folder, exist_ok=True)
    for img_path in img_paths:
        img = Image.open(img_path)
        img_name = os.path.basename(img_path)
        img.save(os.path.join(cluster_folder, img_name), 'TIFF')

# Print the grouped images for verification
for label, img_paths in grouped_images.items():
    print(f'Cluster {label}:')
    for img_path in img_paths:
        print(f'  {img_path}')
