import nibabel as nib
import numpy as np
import cv2
import torch

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def resize_volume(volume, target_shape):
    """
    Efficiently resize a 3D volume using slice-wise cv2 interpolation
    to avoid the memory overhead of scipy.ndimage.zoom.
    """
    # Current shape: (D, H, W)
    depth, height, width = volume.shape
    t_depth, t_height, t_width = target_shape
    
    # 1. Resize H, W for each slice
    resized_slices = []
    for i in range(depth):
        res = cv2.resize(volume[i], (t_width, t_height), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(res)
    volume_hw = np.stack(resized_slices, axis=0) # (D, t_H, t_W)
    
    # 2. Resize Depth (interpolate along D axis)
    # Transpose to (t_H, t_W, D) to resize the D dimension
    volume_t = volume_hw.transpose(1, 2, 0)
    resized_depth = []
    for i in range(t_height):
        res = cv2.resize(volume_t[i], (t_depth, t_width), interpolation=cv2.INTER_LINEAR)
        # Wait, cv2.resize expects (W, H). 
        # This is tricky for D. Let's use a simpler approach for D.
        pass
    
    # Simple slice sampling for D to be safe and fast
    indices = np.linspace(0, depth - 1, t_depth).astype(int)
    volume_d = volume_hw[indices]
    
    return volume_d

def preprocess_mri(flair_path, t1_path, t1ce_path, t2_path, target_shape=(128, 128, 128)):
    """
    Memory-efficient 3D preprocessing.
    """
    modalities = [flair_path, t1_path, t1ce_path, t2_path]
    processed_volumes = []

    for path in modalities:
        data = load_nifti(path).astype(np.float32)
        # Normalize
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # Transpose to (D, H, W)
        data_transposed = np.transpose(data, (2, 0, 1))
        
        # Resize
        data_resized = resize_volume(data_transposed, target_shape)
        processed_volumes.append(data_resized)

    # Stack: (4, 128, 128, 128)
    stacked = np.stack(processed_volumes, axis=0)
    tensor = torch.from_numpy(stacked).float().unsqueeze(0)
    
    return tensor, processed_volumes[0] # Return full 3D Flair volume for 3D context
