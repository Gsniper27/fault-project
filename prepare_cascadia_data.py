import os
import glob
import h5py
import numpy as np
from PIL import Image

# --- Configuration ---
# 1. Set the name of the folder where you will unzip your PNGs.
BASE_DIR = 'tiled_data'

# 2. Define the subdirectories for your different image types.
#    The script REQUIRES 'dem', 'extensional_masks', and 'compressional_masks'.
#    The others ('ctx', 'tw') are optional and will be filled with zeros if not found.
INPUT_DIRS = {
    'dem': os.path.join(BASE_DIR, 'dem'),
    'ctx': os.path.join(BASE_DIR, 'ctx'),
    'tw': os.path.join(BASE_DIR, 'tw'),
    'ext_mask': os.path.join(BASE_DIR, 'extensional_masks'),
    'comp_mask': os.path.join(BASE_DIR, 'compressional_masks')
}

# 3. Define the output filenames.
OUTPUT_H5_FILE = 'cascadia_data_to_mark.h5'
OUTPUT_KEY_FILE = 'cascadia_keys.lst'

# --- End of Configuration ---

def load_image(path, target_size=None, is_mask=False, is_rgb=False):
    """Loads a single image, converts it, and resizes if needed."""
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        if is_rgb:
            img = img.convert('RGB')
        else:
            img = img.convert('L') # Convert to grayscale

        if target_size and img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        arr = np.array(img)
        if is_mask:
            # Normalize mask to be between 0 (black) and 1 (white)
            return arr.astype(np.float32) / 255.0
        else:
            return arr.astype(np.float32)
    except Exception as e:
        print(f"Warning: Could not process image {path}. Error: {e}")
        return None

def main():
    """Main function to process images and create the HDF5 file."""
    print("Starting data preparation...")

    # Find all the primary DEM images to define our samples
    dem_files = glob.glob(os.path.join(INPUT_DIRS['dem'], '*.png'))
    if not dem_files:
        print(f"Error: No PNG files found in '{INPUT_DIRS['dem']}'. Please check your folder setup.")
        return

    keys = []
    
    with h5py.File(OUTPUT_H5_FILE, 'w') as hf:
        for i, dem_path in enumerate(dem_files):
            basename = os.path.basename(dem_path)
            key = os.path.splitext(basename)[0]
            print(f"Processing sample {i+1}/{len(dem_files)}: {key}")

            # Load the required DEM image first
            dem_array = load_image(dem_path)
            if dem_array is None:
                continue

            target_size = dem_array.shape[::-1] # (width, height)

            # --- Load corresponding images and masks ---
            ctx_path = os.path.join(INPUT_DIRS['ctx'], basename)
            tw_path = os.path.join(INPUT_DIRS['tw'], basename)
            ext_mask_path = os.path.join(INPUT_DIRS['ext_mask'], basename)
            comp_mask_path = os.path.join(INPUT_DIRS['comp_mask'], basename)

            ctx_array = load_image(ctx_path, target_size)
            tw_array = load_image(tw_path, target_size, is_rgb=True)
            ext_mask_array = load_image(ext_mask_path, target_size, is_mask=True)
            comp_mask_array = load_image(comp_mask_path, target_size, is_mask=True)

            # --- Create zero arrays for missing optional data ---
            if ctx_array is None:
                ctx_array = np.zeros_like(dem_array)
            if tw_array is None:
                tw_array = np.zeros((*dem_array.shape, 3), dtype=np.uint8)
            if ext_mask_array is None:
                ext_mask_array = np.zeros_like(dem_array)
            if comp_mask_array is None:
                comp_mask_array = np.zeros_like(dem_array)

            # The training script expects distances, not binary masks.
            # For now, we use the masks directly as a placeholder for distance.
            # A value of 0 means "on the fault," and 1 means "far from the fault."
            # The script treats values > 0.35 as "far away".
            te = 1 - ext_mask_array 
            tc = 1 - comp_mask_array

            # Create the group for this sample in the HDF5 file
            group = hf.create_group(key)
            group.create_dataset('CTX', data=ctx_array)
            group.create_dataset('image', data=dem_array) # 'image' is the DEM
            group.create_dataset('TW', data=tw_array.transpose(2,0,1)) # H,W,C -> C,H,W
            group.create_dataset('k0', data=te) # Extensional mask
            group.create_dataset('k1', data=tc) # Compressional mask
            
            keys.append(key)
    
    # Write the list of keys to the key file
    with open(OUTPUT_KEY_FILE, 'w') as f:
        for key in keys:
            f.write(f"{key}\n")

    print(f"\nSuccessfully created HDF5 file: '{OUTPUT_H5_FILE}'")
    print(f"Successfully created key list: '{OUTPUT_KEY_FILE}'")
    print("Data preparation complete!")

if __name__ == '__main__':
    main()