import os

file_path = 'tool.py'

print(f"Fixing TW slicing logic in {file_path}...")

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
fixed_count = 0

for line in lines:
    # Look for the specific slicing pattern causing the error
    if 'TW[:,i0:i0+self.n,j0:j0+self.m]' in line:
        print(f"  -> Fixing line: {line.strip()}")
        # Remove the first colon slice ':, ' to make it work for 2D arrays
        new_line = line.replace('TW[:,i0:i0+self.n,j0:j0+self.m]', 'TW[i0:i0+self.n,j0:j0+self.m]')
        new_lines.append(new_line)
        fixed_count += 1
    elif 'TW[:,::-1,::-1]' in line:
         # Fix other potential 3D slicing for TW if present
         new_line = line.replace('TW[:,::-1,::-1]', 'TW[::-1,::-1]')
         new_lines.append(new_line)
         fixed_count += 1
    elif 'TW[:,::-1]' in line:
         new_line = line.replace('TW[:,::-1]', 'TW[::-1]')
         new_lines.append(new_line)
         fixed_count += 1
    elif 'TW[:,:,::-1]' in line:
         new_line = line.replace('TW[:,:,::-1]', 'TW[:,::-1]')
         new_lines.append(new_line)
         fixed_count += 1
    else:
        new_lines.append(line)

if fixed_count > 0:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"✅ tool.py has been updated! Fixed {fixed_count} instances of incorrect TW slicing.")
else:
    print("⚠️ Warning: Could not find the TW slicing lines to fix. Please check tool.py manually.")