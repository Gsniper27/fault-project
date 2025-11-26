import h5py
import numpy as np

# Path to the file you just uploaded/created
filename = 'cascadia_full_399.h5'

print(f"--- Checking {filename} ---")

try:
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
        count = len(keys)
        print(f"Total tiles found: {count}")
        
        if count == 0:
            print("⚠️ ALERT: The file is EMPTY. The creation script skipped all tiles.")
        else:
            print("✅ SUCCESS: The file contains data!")
            
            # Check the first tile to ensure inner data exists
            first_key = keys[0]
            print(f"Checking structure of first tile: '{first_key}'")
            
            if 'CTX' in f[first_key] and 'target' in f[first_key]:
                print(f"  - CTX shape: {f[first_key]['CTX'].shape}")
                print(f"  - Target shape: {f[first_key]['target'].shape}")
                print("🚀 This file is ready for training!")
            else:
                print("❌ ERROR: Inner datasets (CTX, target) are missing.")

except FileNotFoundError:
    print("Error: Could not find cascadia_full_399.h5. Make sure it is in this folder.")
except Exception as e:
    print(f"An error occurred: {e}")