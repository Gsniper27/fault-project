import h5py
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
OUTPUT_H5_PATH = 'cascadia_full_399.h5'
KEY_LIST_PATH = 'cascadia_keys.lst' 
H, W = 256, 256 # Assuming your tiles are 256x256
DTYPE_INPUT = np.uint8 # Input data usually saved as 8-bit unsigned integer
DTYPE_TARGET = np.float16 # Target masks (distance-to-fault) often saved as floats

# --- RAW DATA FOLDER PATHS (Now ABSOLUTE) ---
# Use the 'r' (raw string) for the main folder path to handle backslashes correctly
RAW_DATA_FOLDER = r'C:\Users\14084\Documents\ArcGIS\Projects\Cascadia_Monkey\fault_project\Raw_Tiles'

# ONLY put the folder NAME here, not the full path.
# The script joins this name to the RAW_DATA_FOLDER automatically.
DEM_FOLDER = 'dem' 

# Ensure these match your actual folder names exactly
COMPRESSIONAL_MASK_FOLDER = 'compressional_masks' 
EXTENSIONAL_MASK_FOLDER = 'extensional_masks'

# --- HELPER FUNCTION: Load a single image ---
# --- HELPER FUNCTION: Load a single image (Highly Robust Version) ---
def load_image_as_array(folder_name, tile_key):
    """
    Loads a single image file by searching the directory for the base key name, 
    regardless of file extension case.
    """
    
    folder_path = os.path.join(RAW_DATA_FOLDER, folder_name)
    
    # 1. Search the directory for a matching file name (case-insensitive base name)
    found_file = None
    # We must iterate over all files in the directory
    for filename in os.listdir(folder_path):
        # We check if the filename starts with the tile key
        if filename.startswith(tile_key):
            # We assume the file is named exactly 'tile_X_Y.ext'
            if filename.split('.')[0] == tile_key:
                found_file = filename
                break
    
    if found_file is None:
        # If no file is found, raise the FileNotFoundError
        raise FileNotFoundError(f"Missing file for tile {tile_key} in {folder_name}")
        
    file_path = os.path.join(folder_path, found_file)
        
    # --- REST OF THE CODE REMAINS THE SAME ---
    # Open the image (assume grayscale 'L' for masks and primary features)
    img = Image.open(file_path).convert('L')
    array = np.array(img, dtype=np.uint8)
    
    # Check dimensions
    if array.shape != (H, W):
        raise ValueError(f"Tile {tile_key} in {folder_name} has incorrect dimensions: {array.shape}")
        
    return array

# --- MAIN DATA LOADING FUNCTIONS ---

def load_raw_features(tile_key):
    """
    Loads all input features (CTX, image, TW) from the DEM folder.
    Assumes DEM is the primary source, replicated for other channels if needed.
    """
    
    # Load the DEM data array (Grayscale HxW)
    dem_array = load_image_as_array(DEM_FOLDER, tile_key)
    
    # 1. CTX (Context Camera / Main Grayscale Feature): Use the DEM directly (HxW)
    CTX = dem_array.astype(DTYPE_INPUT)
    
    # 2. image (RGB / 3-Channel Input): Stack the DEM three times (3xHxW)
    IMAGE = np.stack([dem_array, dem_array, dem_array], axis=0).astype(DTYPE_INPUT)
    
    # 3. TW (Thermal / Auxiliary Feature): Use the DEM directly (HxW)
    TW = dem_array.astype(DTYPE_INPUT)
    
    return CTX, IMAGE, TW

def load_target_masks(tile_key):
    """Loads and combines the two mask files (Extensional and Compressional)."""
    
    # Load the two mask types (Grayscale HxW)
    ext_mask = load_image_as_array(EXTENSIONAL_MASK_FOLDER, tile_key)
    comp_mask = load_image_as_array(COMPRESSIONAL_MASK_FOLDER, tile_key)
    
    # Stack the two masks to form a single target array (2, H, W)
    target_mask = np.stack([ext_mask, comp_mask], axis=0)
    
    # Convert to float type for H5 storage, as targets are often floating-point distance maps
    return target_mask.astype(DTYPE_TARGET)
# --- EXECUTION LOGIC (UPDATED TO FIX KEYERROR) ---
if __name__ == '__main__':
    # 1. Check for the main Raw_Tiles folder
    if not os.path.exists(RAW_DATA_FOLDER):
        print(f"Error: Required top-level folder '{RAW_DATA_FOLDER}' does not exist.")
        exit()

    # 2. Load keys
    try:
        with open(KEY_LIST_PATH, 'r') as f:
            all_tile_keys = [line.strip() for line in f.read().split('\n') if line.strip()]
    except FileNotFoundError:
        print(f"Error: Key list file not found at {KEY_LIST_PATH}")
        exit()
        
    print(f"Loaded {len(all_tile_keys)} keys from the list.")

    # 3. Create the H5 file and write the data
    with h5py.File(OUTPUT_H5_PATH, 'w') as f_out:
        
        for i, key in enumerate(all_tile_keys):
            print(f"Processing tile {i+1}/{len(all_tile_keys)}: {key}")
            
            try:
                # Load data using custom functions
                CTX, IMAGE, TW = load_raw_features(key)
                target = load_target_masks(key) # This is shape (2, 256, 256)
                
                # Create a group for the tile
                tile_group = f_out.create_group(key)
                
                # 1. Save INPUT features
                tile_group.create_dataset('CTX', data=CTX, compression='gzip', dtype=CTX.dtype)
                tile_group.create_dataset('image', data=IMAGE, compression='gzip', dtype=IMAGE.dtype)
                tile_group.create_dataset('TW', data=TW, compression='gzip', dtype=TW.dtype)
                
                # 2. Save TARGETS (Split into specific names to fix KeyError)
                # target[0] is Extensional, target[1] is Compressional
                ext_data = target[0]
                comp_data = target[1]

                # Save as 'tectonics_Extention' AND 'k0' (cover all bases)
                tile_group.create_dataset('tectonics_Extention', data=ext_data, compression='gzip', dtype=ext_data.dtype)
                tile_group.create_dataset('k0', data=ext_data, compression='gzip', dtype=ext_data.dtype)

                # Save as 'tectonics_Compression' AND 'k1' (cover all bases)
                tile_group.create_dataset('tectonics_Compression', data=comp_data, compression='gzip', dtype=comp_data.dtype)
                tile_group.create_dataset('k1', data=comp_data, compression='gzip', dtype=comp_data.dtype)
                
            except FileNotFoundError as e:
                print(f"  ❌ Skipping tile {key}: {e}")
                continue
            except ValueError as e:
                print(f"  ❌ Skipping tile {key} due to data error: {e}")
                continue
            except Exception as e:
                print(f"  ❌ CRITICAL ERROR processing key {key}: {e}. Skipping tile.")
                continue

    print(f"\n✅ H5 File creation complete. Saved to {OUTPUT_H5_PATH}")
