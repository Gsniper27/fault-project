import os

# Path to your tool.py file
file_path = 'tool.py'

print(f"Repairing {file_path}...")

# FIX: Use utf-8 encoding and ignore errors to bypass the Unicode crash
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

new_lines = []
skip = False
repaired = False

for line in lines:
    # 1. Find the anchor line (where mole data is cleaned)
    if 'mole[np.isnan(mole)]=0' in line:
        new_lines.append(line)
        
        # Capture the correct indentation from the anchor line
        indent = line[:line.find('mole')]
        
        # 2. Inject the missing assignment lines
        print("  -> Injecting missing 'te' and 'tc' assignments...")
        new_lines.append(f"{indent}te = f[index]['k0'][:].astype(np.float32)\n")
        new_lines.append(f"{indent}tc = f[index]['k1'][:].astype(np.float32)\n")
        new_lines.append('\n') 
        
        # 3. Start skipping existing lines until we hit the usage line
        # (This prevents duplicate lines if they were just commented out or misplaced)
        skip = True
        repaired = True
        continue
    
    if skip:
        # 4. Stop skipping when we find where 'te' is used
        if 'te[te<0]=999' in line:
            skip = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

if repaired:
    # Save the fixed file with utf-8 encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("✅ tool.py has been successfully repaired!")
else:
    print("⚠️ Warning: Could not find the code block to repair. Please verify tool.py content manually.")