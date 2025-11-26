# tile_data.py
import os
from PIL import Image
import numpy as np
import sys

# --- 1. CONFIGURE YOUR SETTINGS HERE ---

# Paths to your large, aligned map files
INPUT_MAP_PATH = 'input_bathymetry_map.png'  # <--- Make sure this is your bathymetry PNG
MASK_MAP_PATH = 'map.png'      # <--- Make sure this is your fault mask PNG

# Tiling parameters
TILE_SIZE = 256  # The width and height of each square tile (256x256)
STRIDE = 128     # How far to move before cutting the next tile.
                 # (Stride < Tile Size means you get overlapping tiles, which is good for training)

# Output directory
OUTPUT_DIR = 'tiled_data'

# --- END OF CONFIGURATION ---


def tile_maps():
    """
    Cuts the large input and mask maps into smaller, aligned tiles.
    """
    print("Starting the tiling process...")
    print(f"Input Map: {INPUT_MAP_PATH}")
    print(f"Mask Map: {MASK_MAP_PATH}")
    print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}, Stride: {STRIDE}")

    # --- Create output directories ---
    dem_output_dir = os.path.join(OUTPUT_DIR, 'dem')
    mask_output_dir = os.path.join(OUTPUT_DIR, 'extensional_masks')
    os.makedirs(dem_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # --- Load images ---
    try:
        print(f"Loading input map: {INPUT_MAP_PATH}...")
        input_img = Image.open(INPUT_MAP_PATH).convert('L') # Convert to grayscale
        print(f"Loading mask map: {MASK_MAP_PATH}...")
        mask_img = Image.open(MASK_MAP_PATH).convert('L')   # Convert to grayscale
    except FileNotFoundError as e:
        print(f"--- ERROR ---")
        print(f"Error: Could not find a file. Did you save your PNGs with these exact names?")
        print(f"Details: {e}")
        print("Please check your filenames and try again.")
        sys.exit()

    # --- Verify dimensions ---
    if input_img.size != mask_img.size:
        print("--- ERROR ---")
        print("Error: The input map and mask map have different dimensions!")
        print(f"Input map size: {input_img.size} (Width, Height)")
        print(f"Mask map size: {mask_img.size} (Width, Height)")
        print("Please go back to ArcGIS Pro and re-export them with the exact same dimensions.")
        sys.exit()

    print(f"Verified maps are aligned. Dimensions: {input_img.size}")

    input_arr = np.array(input_img)
    mask_arr = np.array(mask_img)
    width, height = input_img.size
    
    tile_count = 0
    skipped_count = 0
    
    # --- Loop through the image and create tiles ---
    print("Slicing maps into tiles...")
    for y in range(0, height - TILE_SIZE + 1, STRIDE):
        for x in range(0, width - TILE_SIZE + 1, STRIDE):
            
            # Extract the tile from the mask array
            mask_tile_arr = mask_arr[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Check if the mask tile contains any faults (is not pure black)
            # This avoids saving thousands of empty ocean tiles.
            # np.sum(mask_tile_arr > 0) checks for any non-black pixels.
            if np.sum(mask_tile_arr) == 0:
                skipped_count += 1
                continue # Skip this tile if it's all black (no faults)

            # If the tile is NOT empty, extract the corresponding tile from the input map
            input_tile_arr = input_arr[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Convert arrays back to images
            input_tile_img = Image.fromarray(input_tile_arr)
            mask_tile_img = Image.fromarray(mask_tile_arr)
            
            # --- Save the tiles ---
            filename = f"tile_{y}_{x}.png"
            input_tile_img.save(os.path.join(dem_output_dir, filename))
            mask_tile_img.save(os.path.join(mask_output_dir, filename))
            tile_count += 1
    
    print("\n--- Tiling Complete! ---")
    print(f"Saved: {tile_count} tiles (containing faults)")
    print(f"Skipped: {skipped_count} empty tiles")
    print(f"Your tiles are ready in the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    # Ensure PIL can handle large images, just in case
    Image.MAX_IMAGE_PIXELS = None 
    tile_maps()