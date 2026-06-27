from skimage.measure import label
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass
from sklearn.neighbors import NearestNeighbors
import torch


def knn_edge_index(coords: np.ndarray, k: int = 5) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    n_points = indices.shape[0]
    src = np.repeat(np.arange(n_points), k)
    dst = indices[:, 1:].flatten()
    edges = np.stack([src, dst], axis=0)
    
    return torch.from_numpy(edges).long()

def calculate_centroids_from_mask(mask_image, min_area=1000, max_features=500, n_colors=32):
    quantized = mask_image.convert("RGB").quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    label_img = np.array(quantized)
    
    all_centroids = []
    
    for class_id in range(n_colors):
        palette_idx = class_id * 3
        if palette_idx + 3 > len(palette):
            continue
        r, g, b = palette[palette_idx:palette_idx + 3]
        if r + g + b < 30:
            continue
        
        binary_mask = (label_img == class_id)
        area = np.sum(binary_mask)
        
        if area < min_area:
            continue
        
        labeled_mask, num_features = label(binary_mask, return_num=True)
        
        if num_features == 0 or num_features > max_features:
            continue
        
        centroids_yx = center_of_mass(binary_mask, labeled_mask, range(1, num_features + 1))
        
        if num_features == 1:
            centroids_yx = [centroids_yx]
        
        for centroid in centroids_yx:
            try:
                y, x = centroid
                if not (np.isnan(x) or np.isnan(y)):
                    all_centroids.append([x, y])
            except (TypeError, ValueError):
                continue
    return np.array(all_centroids) if all_centroids else np.empty((0, 2))
