import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import cdist
from tqdm import tqdm

def load_and_preprocess_images(dataset_dir, dataset_name, feature_mode='resnet'):
    """
    Load all images from a dataset directory and convert to feature vectors.
    
    Args:
        dataset_dir: Directory containing images
        dataset_name: Name of the dataset (for labeling)
        feature_mode: 'resnet' or 'pixels'
    
    Returns:
        images: List of PIL Images
        image_names: List of image filenames
        vectors: numpy array of feature vectors
    """
    print(f"\nLoading images from {dataset_dir}...")
    
    # Get all PNG files
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.png')])
    
    if not files:
        raise FileNotFoundError(f"No PNG files found in {dataset_dir}")
    
    images = []
    image_names = []
    
    # Load images
    for file_name in tqdm(files, desc=f"Loading {dataset_name}"):
        image_path = os.path.join(dataset_dir, file_name)
        try:
            img = Image.open(image_path).convert('RGB')
            images.append(img)
            image_names.append(file_name)
        except Exception as e:
            print(f"Warning: Could not load {file_name}: {e}")
            continue
    
    print(f"Loaded {len(images)} images")
    
    # Convert images to feature vectors
    if feature_mode == 'resnet':
        print(f"Converting images to feature vectors using ResNet...")
        vectors = images_to_resnet_features(images)
    elif feature_mode == 'pixels':
        print(f"Converting images to feature vectors using pixel values...")
        vectors = images_to_pixel_features(images)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}. Use 'resnet' or 'pixels'")
    
    return images, image_names, vectors

def images_to_resnet_features(images):
    """
    Convert images to feature vectors using pre-trained ResNet.
    
    Args:
        images: List of PIL Images
    
    Returns:
        numpy array of feature vectors (N x 2048 for ResNet-50)
    """
    # Load pre-trained ResNet-50
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    
    # Remove the final classification layer to get features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting ResNet features"):
            # Preprocess image
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract features
            feat = model(img_tensor)
            feat = feat.squeeze().cpu().numpy().flatten()
            features.append(feat)
    
    return np.array(features)

def images_to_pixel_features(images, resize_to=(224, 224)):
    """
    Convert images to feature vectors using raw pixel values.
    
    Args:
        images: List of PIL Images
        resize_to: Target size for resizing (to ensure consistent dimensions)
    
    Returns:
        numpy array of feature vectors (N x (H*W*3))
    """
    features = []
    
    for img in tqdm(images, desc="Extracting pixel features"):
        # Resize to consistent size
        img_resized = img.resize(resize_to)
        
        # Convert to numpy array and flatten
        img_array = np.array(img_resized)  # Shape: (H, W, 3)
        img_flat = img_array.flatten()  # Shape: (H*W*3,)
        
        # Normalize to [0, 1]
        img_flat = img_flat.astype(np.float32) / 255.0
        
        features.append(img_flat)
    
    return np.array(features)

def visualize_embeddings(all_vectors, all_labels, all_dataset_names, method='umap', output_path='dataset_embeddings.png'):
    """
    Project vectors to 2D using PCA/UMAP and visualize.
    
    Args:
        all_vectors: numpy array of all feature vectors (N x D)
        all_labels: list of labels (image names)
        all_dataset_names: list of dataset names for each vector
        method: 'pca', 'umap', or 'both'
        output_path: path to save the visualization
    """
    print(f"\nProjecting to 2D using {method}...")
    
    if method == 'pca' or method == 'both':
        # PCA projection
        pca = PCA(n_components=2, random_state=42)
        coords_pca = pca.fit_transform(all_vectors)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    if method == 'umap' or method == 'both':
        # UMAP projection
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords_umap = reducer.fit_transform(all_vectors)
    
    # Define colors for each dataset
    dataset_colors = {
        'original': '#1f77b4',  # Blue
        'LEGO': '#ff7f0e',     # Orange
        'Pop_Art': '#2ca02c'    # Green
    }
    
    # Create figure
    if method == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        ax2 = None
    
    # Plot PCA
    if method == 'pca' or method == 'both':
        ax = ax1 if method == 'both' else ax1
        for dataset_name in ['original', 'LEGO', 'Pop_Art']:
            mask = np.array(all_dataset_names) == dataset_name
            if mask.sum() > 0:
                coords = coords_pca if method == 'pca' or method == 'both' else coords_umap
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c=dataset_colors[dataset_name], 
                          label=dataset_name, 
                          alpha=0.6, 
                          s=50)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('PCA Projection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot UMAP
    if method == 'umap' or method == 'both':
        ax = ax2 if method == 'both' else ax1
        for dataset_name in ['original', 'LEGO', 'Pop_Art']:
            mask = np.array(all_dataset_names) == dataset_name
            if mask.sum() > 0:
                coords = coords_umap
                ax.scatter(coords[mask, 0], coords[mask, 1], 
                          c=dataset_colors[dataset_name], 
                          label=dataset_name, 
                          alpha=0.6, 
                          s=50)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title('UMAP Projection', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

def analyze_clustering(all_vectors, all_dataset_names, all_labels):
    """
    Analyze whether images cluster more by dataset or by image identity.
    
    Args:
        all_vectors: numpy array of all feature vectors
        all_dataset_names: list of dataset names for each vector
        all_labels: list of image filenames for each vector
    """
    print("\n" + "="*60)
    print("Clustering Analysis")
    print("="*60)
    
    # Extract base image names from filenames (remove dataset suffix)
    base_names = []
    for filename in all_labels:
        # Extract base name (e.g., "1" from "1.png" or "1_LEGO.png")
        base = filename.replace('_LEGO.png', '').replace('_Pop_Art.png', '').replace('.png', '')
        base_names.append(base)
    
    # Calculate average distance within same dataset vs across datasets
    distances = cdist(all_vectors, all_vectors, metric='cosine')
    
    # Separate all combinations
    same_image_same_style = []      # Image 1 Original vs Image 1 Original (shouldn't exist, but just in case)
    same_image_different_styles = [] # Image 1 Original vs Image 1 LEGO, Image 1 LEGO vs Image 1 Pop Art
    different_images_same_style = [] # Image 1 Original vs Image 2 Original
    different_images_different_styles = [] # Image 1 Original vs Image 2 LEGO
    
    for i in range(len(all_vectors)):
        for j in range(i+1, len(all_vectors)):
            dist = distances[i, j]
            
            same_image = (base_names[i] == base_names[j])
            same_style = (all_dataset_names[i] == all_dataset_names[j])
            
            if same_image and same_style:
                same_image_same_style.append(dist)
            elif same_image and not same_style:
                same_image_different_styles.append(dist)
            elif not same_image and same_style:
                different_images_same_style.append(dist)
            else:  # not same_image and not same_style
                different_images_different_styles.append(dist)
    
    # For backward compatibility, also compute the old metrics
    within_dataset_dists = same_image_same_style + different_images_same_style
    across_dataset_dists = same_image_different_styles + different_images_different_styles
    within_image_dists = same_image_different_styles
    across_image_dists = different_images_same_style
    
    print(f"\nDistance Analysis (Cosine Distance):")
    print(f"\nAll 4 combinations:")
    if len(same_image_different_styles) > 0:
        print(f"  1. Same image, different styles (e.g., Img1 Original vs Img1 LEGO):")
        print(f"     mean={np.mean(same_image_different_styles):.4f}, std={np.std(same_image_different_styles):.4f}")
    if len(different_images_same_style) > 0:
        print(f"  2. Different images, same style (e.g., Img1 Original vs Img2 Original):")
        print(f"     mean={np.mean(different_images_same_style):.4f}, std={np.std(different_images_same_style):.4f}")
    if len(different_images_different_styles) > 0:
        print(f"  3. Different images, different styles (e.g., Img1 Original vs Img2 LEGO):")
        print(f"     mean={np.mean(different_images_different_styles):.4f}, std={np.std(different_images_different_styles):.4f}")
    
    print(f"\nInterpretation:")
    if len(same_image_different_styles) > 0 and len(different_images_same_style) > 0:
        if np.mean(same_image_different_styles) < np.mean(different_images_same_style):
            print(f"  → CONTENT matters MORE: Same image across styles ({np.mean(same_image_different_styles):.4f})")
            print(f"     is more similar than different images in same style ({np.mean(different_images_same_style):.4f})")
        else:
            print(f"  → STYLE matters MORE: Different images in same style ({np.mean(different_images_same_style):.4f})")
            print(f"     is more similar than same image across styles ({np.mean(same_image_different_styles):.4f})")
    
    if len(same_image_different_styles) > 0 and len(different_images_different_styles) > 0:
        print(f"  → Style effect: Same image across styles ({np.mean(same_image_different_styles):.4f}) vs")
        print(f"     Different images different styles ({np.mean(different_images_different_styles):.4f})")
        print(f"     Difference: {np.mean(different_images_different_styles) - np.mean(same_image_different_styles):.4f}")
    else:
        print(f"\n  Within same image (diff styles): No pairs found (all images may be from same dataset)")
        print(f"  Across different images:          mean={np.mean(across_image_dists):.4f}, std={np.std(across_image_dists):.4f}")
    
    print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize dataset embeddings using PCA/UMAP')
    parser.add_argument('--original_dir', type=str, default='dataset_Kodack24',
                       help='Directory containing original images')
    parser.add_argument('--lego_dir', type=str, default='dataset_Kodack24_LEGO',
                       help='Directory containing LEGO style images')
    parser.add_argument('--popart_dir', type=str, default='dataset_Kodack24_Pop_Art',
                       help='Directory containing Pop Art style images')
    parser.add_argument('--method', type=str, default='both', choices=['pca', 'umap', 'both'],
                       help='Dimensionality reduction method')
    parser.add_argument('--feature_mode', type=str, default='resnet', choices=['resnet', 'pixels'],
                       help='Feature extraction mode: resnet (semantic features) or pixels (raw pixel values)')
    parser.add_argument('--output', type=str, default='dataset_embeddings.png',
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Load all datasets
    all_vectors = []
    all_labels = []
    all_dataset_names = []
    
    # Original dataset
    if os.path.exists(args.original_dir):
        images, names, vectors = load_and_preprocess_images(args.original_dir, 'original', args.feature_mode)
        all_vectors.append(vectors)
        all_labels.extend(names)
        all_dataset_names.extend(['original'] * len(names))
    else:
        print(f"Warning: {args.original_dir} not found, skipping...")
    
    # LEGO dataset
    if os.path.exists(args.lego_dir):
        images, names, vectors = load_and_preprocess_images(args.lego_dir, 'LEGO', args.feature_mode)
        all_vectors.append(vectors)
        all_labels.extend(names)
        all_dataset_names.extend(['LEGO'] * len(names))
    else:
        print(f"Warning: {args.lego_dir} not found, skipping...")
    
    # Pop Art dataset
    if os.path.exists(args.popart_dir):
        images, names, vectors = load_and_preprocess_images(args.popart_dir, 'Pop_Art', args.feature_mode)
        all_vectors.append(vectors)
        all_labels.extend(names)
        all_dataset_names.extend(['Pop_Art'] * len(names))
    else:
        print(f"Warning: {args.popart_dir} not found, skipping...")
    
    if not all_vectors:
        raise ValueError("No datasets found! Check directory paths.")
    
    # Concatenate all vectors
    all_vectors = np.vstack(all_vectors)
    print(f"\nTotal vectors: {len(all_vectors)}")
    print(f"Vector dimension: {all_vectors.shape[1]}")
    
    # Analyze clustering
    analyze_clustering(all_vectors, all_dataset_names, all_labels)
    
    # Visualize
    visualize_embeddings(all_vectors, all_labels, all_dataset_names, 
                        method=args.method, output_path=args.output)
    
    print(f"\nDone! Visualization saved to {args.output}")

if __name__ == '__main__':
    main()

