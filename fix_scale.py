import os

file_path = 'tool.py'

print(f"Fixing scaling logic in {file_path}...")

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
fixed = False

for line in lines:
    # Find the line that calculates the random scale
    if 'scale = np.random.rand()*0.5+0.75' in line:
        print("  -> Found scaling line. Modifying to prevent shrinking...")
        # Change range from [0.75, 1.25] to [1.0, 1.5]
        # This ensures the image is never smaller than the crop size (256x256)
        new_line = line.replace('0.75', '1.0')
        new_lines.append(new_line)
        fixed = True
    else:
        new_lines.append(line)

if fixed:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("✅ tool.py has been updated! Random zooming will no longer shrink images.")
else:
    print("⚠️ Warning: Could not find the scaling line to fix. Please check tool.py manually.")