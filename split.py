import os
import random
import shutil

# Set your main directory containing 1000 CIF files
source_folder = './cif-files'
train_folder = os.path.join(source_folder, 'train')
test_folder = os.path.join(source_folder, 'test')

# Create train and test directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get all .cif files
cif_files = [f for f in os.listdir(source_folder) if f.endswith('.cif')]

# Shuffle and split
random.shuffle(cif_files)
split_index = int(0.8 * len(cif_files))
train_files = cif_files[:split_index]
test_files = cif_files[split_index:]

# Move files
for f in train_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))
for f in test_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(test_folder, f))

print(f"Moved {len(train_files)} files to train/, {len(test_files)} files to test/")
