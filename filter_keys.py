import h5py
import numpy as np

def inspect_file(file_path):
    print(f"\n--- Inspecting File: {file_path} ---")
    try:
        with h5py.File(file_path, 'r') as f:
            h5_keys = list(f.keys())
            
            if h5_keys:
                print("First 20 keys:")
                print(h5_keys[:20]) 
                print(f"\nTotal keys found: {len(h5_keys)}")
            else:
                print("The file appears to be empty or has no top-level keys.")
                
            # Check for specific keys expected in the original data file
            if 'CTX' in h5_keys or 'image' in h5_keys:
                print("\nNOTE: This file looks like a large, non-tiled dataset, not individual tiles.")
            elif any(key.startswith('tile_') for key in h5_keys[:20]):
                print("\n✅ This file CONTAINS TILE KEYS and is likely the correct structure!")


    except FileNotFoundError:
        print(f"ERROR: Could not find the file at {file_path}. Please ensure it is in the same folder as this script.")
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}")

# Run inspection on both new files
inspect_file('cascadia_tiles.h5')
inspect_file('cascadia_all.h5')