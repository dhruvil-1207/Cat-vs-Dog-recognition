import os
import shutil
import random

# Adjust source if your images are one level deeper (e.g., ../data/train/train)
SOURCE = '../data/train/' 
BASE_DIR = '../dataset_final/'

def setup():
    for split in ['train', 'test']:
        for animal in ['cats', 'dogs']:
            os.makedirs(os.path.join(BASE_DIR, split, animal), exist_ok=True)

    all_images = [f for f in os.listdir(SOURCE) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_images)

    # 80% Train, 20% Test
    split_idx = int(len(all_images) * 0.8)
    train_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    def move_to_dest(files, folder):
        for f in files:
            label = 'cats' if f.startswith('cat') else 'dogs'
            shutil.copy(os.path.join(SOURCE, f), os.path.join(BASE_DIR, folder, label, f))

    print("Organizing 25,000 images...")
    move_to_dest(train_files, 'train')
    move_to_dest(test_files, 'test')
    print("Done! Check your dataset_final folder.")

if __name__ == "__main__":
    setup()