import h5py

H5_FILE_PATH = 'H5/all_eqA_296_tomark.h5'
KEY_LIST_PATH = 'cascadia_keys.lst'
OUTPUT_PATH = 'keys_filtered.lst'

# 1. Get keys from the H5 file
try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        h5_keys = set(f.keys())
except Exception as e:
    print(f"Error reading H5 file: {e}")
    exit()

# 2. Get keys from the training list file
with open(KEY_LIST_PATH, 'r') as f:
    # Note: We need to handle the list formatting from the original file
    # The key list files you provided (like cascadia_keys.lst) 
    # have lines that need to be cleaned up.
    list_keys = [line.strip() for line in f.read().split('\n') if line.strip() and not line.startswith('[')]

# 3. Find the valid keys (the intersection)
valid_keys = sorted(list(h5_keys.intersection(set(list_keys))))

# 4. Save the valid keys to a new file
with open(OUTPUT_PATH, 'w') as f:
    for key in valid_keys:
        f.write(key + '\n')

print(f"Filter complete. Found {len(valid_keys)} valid keys.")
print(f"Saved to {OUTPUT_PATH}")