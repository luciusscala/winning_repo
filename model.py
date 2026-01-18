import os
# Set HuggingFace cache to workspace directory (fixes Jupyter permission errors)
# Use current working directory instead of home to avoid permission issues
workspace_cache = os.path.join(os.getcwd(), ".hf_cache")
hub_cache = os.path.join(workspace_cache, "hub")
os.makedirs(workspace_cache, exist_ok=True)
os.makedirs(hub_cache, exist_ok=True)

# Set all possible cache environment variables BEFORE any imports
os.environ['HF_HOME'] = workspace_cache
os.environ['HF_DATASETS_CACHE'] = workspace_cache
os.environ['HUGGINGFACE_HUB_CACHE'] = hub_cache
os.environ['HF_HUB_CACHE'] = hub_cache
os.environ['TRANSFORMERS_CACHE'] = workspace_cache
# Override system cache locations
os.environ['XDG_CACHE_HOME'] = workspace_cache

import tensorflow as tf
from tensorflow.keras import layers
import argparse
from datasets import load_dataset
from PIL import Image
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, image_dim=None, output_scale=3.0):
        super().__init__()
        self.image_dim = image_dim  # Will be set dynamically
        self.conv1_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.conv1_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.bn1 = layers.BatchNormalization()
        self.conv2_1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.bn2 = layers.BatchNormalization()
        self.conv3_1 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.bn3 = layers.BatchNormalization()
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.4)
        self.output_layer = layers.Dense(3, activation="tanh")
        self.output_scale = output_scale
    
    def call(self, x, training=False):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.bn1(x, training=training)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x) * self.output_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('-b', '--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    if args.command == 'train':
        # Load dataset (cache already configured at top of file)
        print("Loading dataset...")
        print(f"Using cache directory: {workspace_cache}")
        print(f"Using hub cache: {hub_cache}")
        
        # Try loading with explicit cache_dir parameter
        ds = load_dataset("imageomics/sentinel-beetles", cache_dir=workspace_cache)
        
        # Find most common image size
        print("Analyzing image sizes in dataset...")
        sizes = {}
        sample_size = min(1000, len(ds['train']))  # Check up to 1000 images
        for i, example in enumerate(ds['train']):
            if i >= sample_size:
                break
            try:
                img = example['file_path']
                if isinstance(img, list):
                    img = img[0]
                if isinstance(img, str):
                    img = Image.open(img)
                size = img.size
                sizes[size] = sizes.get(size, 0) + 1
            except:
                continue
        
        if not sizes:
            print("ERROR: Could not read any images!")
            exit(1)
        
        # Get most common size
        most_common_size = max(sizes.items(), key=lambda x: x[1])
        target_size = most_common_size[0]
        count = most_common_size[1]
        print(f"Most common size: {target_size} ({count} out of {sample_size} samples)")
        print(f"Top 5 sizes: {sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # Filter for most common size
        print(f"Filtering for {target_size} images...")
        def is_target_size(example):
            try:
                img = example['file_path']
                if isinstance(img, list):
                    img = img[0]
                if isinstance(img, str):
                    img = Image.open(img)
                return img.size == target_size
            except:
                return False
        
        ds_filtered = ds['train'].filter(is_target_size)
        print(f"Filtered dataset size: {len(ds_filtered)}")
        
        if len(ds_filtered) == 0:
            print("ERROR: No images found after filtering!")
            exit(1)
        
        # Process examples
        def process_example(example):
            img = example['file_path']
            if isinstance(img, list):
                img = img[0]
            if isinstance(img, str):
                img = Image.open(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Just normalize (no resize needed)
            img_array = np.array(img, dtype=np.float32) / 255.0
            x = tf.convert_to_tensor(img_array, dtype=tf.float32)
            y = tf.stack([example['SPEI_30d'], example['SPEI_1y'], example['SPEI_2y']])
            return x, tf.cast(y, tf.float32)
        
        # Create dataset
        def generator():
            for example in ds_filtered:
                yield process_example(example)
        
        # Get actual image size from first example
        first_example = next(iter(ds_filtered))
        img = first_example['file_path']
        if isinstance(img, list):
            img = img[0]
        if isinstance(img, str):
            img = Image.open(img)
        actual_size = img.size
        print(f"Using image size: {actual_size}")
        
        output_signature = (
            tf.TensorSpec(shape=(actual_size[1], actual_size[0], 3), dtype=tf.float32),  # height, width, channels
            tf.TensorSpec(shape=(3,), dtype=tf.float32)
        )
        
        tf_ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        tf_ds = tf_ds.shuffle(1000).batch(args.batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        
        # Create and train model
        model = Model(image_dim=actual_size, output_scale=3.0)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        print("Starting training...")
        model.fit(tf_ds, epochs=30)
