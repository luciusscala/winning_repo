import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

import numpy as np

import argparse

from datasets import load_dataset

class Model(tf.keras.Model):
    def __init__(self,image_dim=256,output_scale=3.0):
        super(Model, self).__init__()

        # Assume images are square for now
        image_width = image_dim
        image_height = image_dim

        # Block 1: (Conv(3×3) → Conv(3×3) → MaxPool) 32 filters
        num_filters = 32
        self.conv1_1 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.conv1_2 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D((2,2))
        self.bn1 = layers.BatchNormalization()

        # Block 2: (Conv(3×3) → Conv(3×3) → MaxPool) 64 filters
        num_filters = 64
        self.conv2_1 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D((2,2))
        self.bn2 = layers.BatchNormalization()

        # Block 3: (Conv(3×3) → Conv(3×3) → MaxPool) 128 filters
        num_filters = 128
        self.conv3_1 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D((2,2))
        self.bn3 = layers.BatchNormalization()

        # Head
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(num_filters, activation='relu')
        self.dropout = layers.Dropout(0.4)

        # Output
        self.output_layer = layers.Dense(3, activation="tanh")
        self.output_scale = output_scale
    
    def call(self, x, training=False):
        # Call block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.bn1(x, training=training)

        # Call block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)

        # Call block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)

        # Head
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)

        # Return Output
        return self.output_layer(x) * self.output_scale

# Formatting for training dataset
def ds_to_fit_format(row):
    x = np.array(row['tensors'], dtype=np.float32) / 255.0
    y = np.array([row['SPEI_30d'], row['SPEI_1y'], row['SPEI_2y']], dtype=np.float32)

    return {'x': x, 'y': y}

# Training
def train(args, model, ds):
    # Format dataset for training
    # print(f"b4 transform {ds['train']['tensors'].shape}")
    ds['train'].set_transform(ds_to_fit_format)
    # print(f"b4 to_tf_dataset {training_ds[0]['tensors'].shape}")
    tf_ds = ds['train'].to_tf_dataset(columns='x', label_cols='y', batch_size=args.batch_size, shuffle=True)
    print("post to_tf_dataset")

    for batch_x, batch_y in tf_ds.take(1):
        print(batch_x.shape)
        print(batch_y.shape)

    # Perform training
    model.fit(tf_ds, batch_size=args.batch_size, epochs=30)

# Hub for doing things w/ model
def main(args):
    img_size = 256
    model = Model(image_dim=img_size,output_scale=3.0)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # model.build(input_shape=(None, img_size, img_size, 3))

    if hasattr(args, 'command') and args.command == 'train':
        train(args, model, load_dataset("./dataset"))

    return 0

# Define command-line arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='command')

# Train subcommand
train_parser = subparsers.add_parser('train')
train_parser.add_argument('-b', "--batch-size", type=int, default=32, help="number of datapoints to train on for each iteration")

# Collect command-line arguments
args = parser.parse_args()

# Call main
if (__name__ == "__main__"):
    main(args)