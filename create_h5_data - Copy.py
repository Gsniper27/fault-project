import h5py
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_H5_PATH = 'cascadia_full_399.h5'
KEY_LIST_PATH = 'cascadia_keys.lst' # Contains 399 tile_X_Y keys
RAW_DATA_FOLDER = 'Raw_Tiles/'      # Create a folder for your PNGs and mask data

# --- PLACEHOLDER FUNCTIONS (YOU MUST FILL THESE IN) ---
def load_raw_features(tile_key):
    """Loads the input image features (CTX, image, TW) for a given tile."""
    
    # Placeholder: Replace this logic with code to load your 
    # tile_X_Y.png (for CTX/image) and related data.
    
    # Assuming all input features are 256x256 grayscale or 3-channel images:
    H, W = 256, 256
    
    # Example: Loading CTX (Context, assumed grayscale)
    ctx_data = np.random.rand(H, W) * 255.0 # Replace with actual image loading (e.g., PIL/OpenCV)
    
    # Example: Loading image (MOLA/Visual, assumed grayscale or RGB)
    image_data = np.random.rand(3, H, W) * 255.0 

    # Example: Loading TW (Thermal/Auxiliary, assumed grayscale)
    tw_data = np.random.rand(H, W) * 100.0
    
    # The arrays must have the same shape for all tiles!
    return ctx_data.astype(np.uint8), image_data.astype(np.uint8), tw_data.astype(np.uint8)


def load_target_masks(tile_key):
    """Loads the ground truth mask (the label) for a given tile."""
    
    # Placeholder: You need the fault masks that correspond to the image tiles.
    # This data is usually derived from your original fault shapefiles.
    
    H, W = 256, 256
    # Example: Target/Mask data (assumed two channels: Extention and Compression)
    # This array MUST contain the fault/mask data.
    target_mask = np.zeros((2, H, W), dtype=np.float16) 
    
    return target_mask

# --- MAIN CREATION LOGIC ---
if __name__ == '__main__':
    # 1. Load all 399 tile keys from your list
    try:
        with open(KEY_LIST_PATH, 'r') as f:
            all_tile_keys = [line.strip() for line in f.read().split('\n') if line.strip()]
    except FileNotFoundError:
        print(f"Error: Key list file not found at {KEY_LIST_PATH}")
        exit()
        
    print(f"Loaded {len(all_tile_keys)} keys from the list.")

    if len(all_tile_keys) != 399:
        print(f"Warning: Expected 399 keys, found {len(all_tile_keys)}. Proceeding with available keys.")

    # 2. Create the H5 file and write the data
    with h5py.File(OUTPUT_H5_PATH, 'w') as f_out:
        
        for i, key in enumerate(all_tile_keys):
            print(f"Processing tile {i+1}/{len(all_tile_keys)}: {key}")
            
            try:
                # Load data using your custom functions
                ctx, image, tw = load_raw_features(key)
                target = load_target_masks(key)
                
                # Create a group for the tile
                tile_group = f_out.create_group(key)
                
                # Save the input features and the target masks as datasets
                tile_group.create_dataset('CTX', data=ctx, compression='gzip')
                tile_group.create_dataset('image', data=image, compression='gzip')
                tile_group.create_dataset('TW', data=tw, compression='gzip')
                tile_group.create_dataset('target', data=target, compression='gzip') # Target masks (fault/label data)
                
            except Exception as e:
                print(f"CRITICAL ERROR processing key {key}: {e}. Skipping tile.")
                continue

    print(f"\n✅ H5 File creation complete. Saved to {OUTPUT_H5_PATH}")