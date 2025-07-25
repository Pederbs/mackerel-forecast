import os
import time
from tqdm import tqdm
import datetime
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt

from dataset_manager import DatasetManager
from models import Generator, Discriminator
from losses import (
    generator_loss,
    discriminator_loss
)
from utils.logger import logger
from utils.plotting import generate_unraveled_image, show_predictions

# Dataset paths
# DATASET_PATH = "/home/peder/fish-forecast/Datasets/fa_mac_may_aug_50p_catch_continuous_GAT"
DATASET_PATH = "/home/anna/msc_oppgave/fish-forecast/Datasets/fa_mac_may_aug_50p_catch_continuous_GAT"
CHANNEL_PREFIXES = {
    f"{DATASET_PATH}/chl/0/": "chl_",
    # f"{DATASET_PATH}/chl/3/": "chl_",
    # f"{DATASET_PATH}/chl/10/": "chl_",
    # f"{DATASET_PATH}/chl/16/": "chl_",
    # f"{DATASET_PATH}/chl/25/": "chl_",
    f"{DATASET_PATH}/o2/0/": "o2_",
    # f"{DATASET_PATH}/o2/3/": "o2_",
    # f"{DATASET_PATH}/o2/10/": "o2_",
    # f"{DATASET_PATH}/o2/16/": "o2_",
    # f"{DATASET_PATH}/o2/25/": "o2_",
    f"{DATASET_PATH}/phyc/0/": "phyc_",
    # f"{DATASET_PATH}/phyc/3/": "phyc_",
    # f"{DATASET_PATH}/phyc/10/": "phyc_",
    # f"{DATASET_PATH}/phyc/16/": "phyc_",
    # f"{DATASET_PATH}/phyc/25/": "phyc_",
    f"{DATASET_PATH}/zooc/0/": "zooc_",
    # f"{DATASET_PATH}/zooc/3/": "zooc_",
    # f"{DATASET_PATH}/zooc/10/": "zooc_",
    # f"{DATASET_PATH}/zooc/16/": "zooc_",
    # f"{DATASET_PATH}/zooc/25/": "zooc_",
    f"{DATASET_PATH}/salinity/0/": "sal_h0_",
    # f"{DATASET_PATH}/salinity/3/": "sal_h3_",
    # f"{DATASET_PATH}/salinity/10/": "sal_h10_",
    # f"{DATASET_PATH}/salinity/15/": "sal_h15_",
    # f"{DATASET_PATH}/salinity/25/": "sal_h25_",
    f"{DATASET_PATH}/temperature/0/": "temp_h0_",
    # f"{DATASET_PATH}/temperature/3/": "temp_h3_",
    # f"{DATASET_PATH}/temperature/10/": "temp_h10_",
    # f"{DATASET_PATH}/temperature/15/": "temp_h15_",
    # f"{DATASET_PATH}/temperature/25/": "temp_h25_",
}
OUTPUT_DIR = f"{DATASET_PATH}/fiskdir/"
OUTPUT_PREFIX = "catch_"

# Hyperparameters
#--------------------------
# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 512
INPUT_CHANNELS = len(CHANNEL_PREFIXES)
OUTPUT_CHANNELS = 3

# Dataset parameters
TEST_SIZE = 0.3         # 30% of the dataset will be used for testing
RANDOM_STATE = 42       # Random state for reproducibility
SHUFFLE_BUFFER = 1000   # Buffer size for shuffling the dataset

# Augmentation parameters
AUGMENTATION = True     # Use data augmentation
JITTER_PADDING = 100    # (px) Expand image size by this amount for random jitter and later crop

# Training parameters
GEN_LEARNING_RATE = 2e-4    # Learning rate for generator
GEN_BETA_1 = 0.5            # Beta 1 parameter for Adam optimizer (generator)

DISC_LEARNING_RATE = 1e-3   # Learning rate for discriminator
DISC_BETA_1 = 0.5           # Beta 1 parameter for Adam optimizer (discriminator)

LAMBDA = 10.0           # Lambda parameter for generator loss 

BATCH_SIZE = 4          # Cannot be less than number of replicas, (i.e. 2)
STEPS = 10_000          # Number of training steps

# Checkpointing and logging
LOG_DIR = f'validation_logs'

os.makedirs(LOG_DIR, exist_ok=True)
# -------------------------


# 0) Multi-GPU setup
tf.keras.backend.set_floatx('float32')
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(["GPU:0"])
logger.info(f'Running on {strategy.num_replicas_in_sync} replicas')

if BATCH_SIZE % strategy.num_replicas_in_sync != 0:
    logger.error(f'Ensure that BATCH size is devisable by number of replicas: ({BATCH_SIZE} / {strategy.num_replicas_in_sync})')
    exit()

# 1) Prepare datasets (outside scope)
dirs = list(CHANNEL_PREFIXES.keys())
manager = DatasetManager(
    input_dirs=dirs,
    channel_prefixes=CHANNEL_PREFIXES,
    output_dir=OUTPUT_DIR,
    output_prefix=OUTPUT_PREFIX,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    data_augmentation=AUGMENTATION,
    jitter_padding=JITTER_PADDING
)
# Preconditioning: check if all images are present and of the same size
validation_passed = manager.check_all_image_shapes()
if not validation_passed:
    raise ValueError("Dataset validation failed")

raw_train_ds, raw_test_ds = manager.create_train_test_datasets(
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER
)

# Create finite set and distribute it
finite_train_ds = raw_train_ds.repeat().take(STEPS+1)
dist_train_ds = strategy.experimental_distribute_dataset(finite_train_ds)


# iterate over dist_train_ds and log all unsuccessful images
train_iter = iter(dist_train_ds)

bar = tqdm(
    enumerate(train_iter),
    total=STEPS,
    desc="Training",
    unit="step",
    leave=True
)

for step, inputs in bar:

    # try to get the next batch if it fails, log the date with the error to one file, also save the tensor to a file, named the date and the variable
    try:
        inp, re = inputs

    except Exception as e:
        logger.error(f"Error in batch {i}: {e}")
        # Save the batch to a file
        with open(f"{LOG_DIR}/failed_batches.txt", "a") as f:
            f.write(f"Batch {step}: {e}\n")
        # Save the tensor to a file
        tf.io.write_file(f"{LOG_DIR}/failed_batch_{step}.nc", tf.io.serialize_tensor(inputs))
        continue

    try:
        # assert that both inp and re are the correct shape
        if inp.shape == (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS):
            pass

    except Exception as e:
        # Log the error
        logger.error(f"Error in batch {step}: {e}")
        # Save the batch to a file
        with open(f"{LOG_DIR}/failed_batches.txt", "a") as f:
            f.write(f"Batch {step}: {e}\n")
        # Save the tensor to a file
        tf.io.write_file(f"{LOG_DIR}/failed_batch_{step}.nc", tf.io.serialize_tensor(inp))
        continue

    try:
        if re.shape == (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS):
            pass

    except Exception as e:
        # Log the error
        logger.error(f"Error in batch {step}: {e}")
        # Save the batch to a file
        with open(f"{LOG_DIR}/failed_batches.txt", "a") as f:
            f.write(f"Batch {step}: {e}\n")
        # Save the tensor to a file
        tf.io.write_file(f"{LOG_DIR}/failed_batch_{step}.nc", tf.io.serialize_tensor(re))
        continue