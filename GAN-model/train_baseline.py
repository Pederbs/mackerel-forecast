"""
training script for the baseline model to avoid having to change hyperparameters.
"""

import os
import time
import math
from tqdm import tqdm
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset_manager import DatasetManager
from models import Generator, Discriminator
from losses import generator_loss, discriminator_loss
from utils.logger import logger
from utils.plotting import generate_unraveled_image, show_predictions
from utils.hyperparams import save_hyperparameters, collect_hyperparameters
from evaluate import ImageEvaluator, TrainingEvaluator

tf.keras.backend.clear_session()

# -------------------------
# Paths & Parameters
# -------------------------
# Dataset paths
DATASET_NAME = 'balanced_big'  # Name of the dataset
DATASET_PATH = f"/home/peder/fish-forecast/Datasets/{DATASET_NAME}/"


# Checkpointing and logging
# ITERATION_NAME = 'test-area_19to24_60p_LCO_TsOz_fixedDS'  # Name for this training iteration
ITERATION_NAME = f'{DATASET_NAME}_baseline_last100'  # Name for this training iteration
# DATASET_PATH = "/home/anna/msc_oppgave/fish-forecast/Datasets/fa_mac_may_aug_50p_catch_continuous_GAT"
CHANNEL_PREFIXES = {
    f"{DATASET_PATH}/no3/3.0/": "",
    f"{DATASET_PATH}/o2/3.0/": "",
    f"{DATASET_PATH}/zooc/3.0/": "",
    # f"{DATASET_PATH}/no3/29.0/": "",
    # f"{DATASET_PATH}/o2/29.0/": "",
    # f"{DATASET_PATH}/zooc/22.0/": "",
    f"{DATASET_PATH}/thetao/29_44/": "",
    f"{DATASET_PATH}/uo/1_54/": "",
}
OUTPUT_DIR = f"{DATASET_PATH}/catch/"
OUTPUT_PREFIX = "catch_"


# Hyperparameters
#--------------------------
# Image parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_CHANNELS = len(CHANNEL_PREFIXES)
OUTPUT_CHANNELS = 3

# Dataset parameters
VALIDATION_SIZE = 0.1   # 10% of total dataset for validation
TEST_SIZE = 0.1         # 10% of total dataset for test predictions during training
RANDOM_STATE = 42       # Random state for reproducibility
SHUFFLE_BUFFER = 1000   # Buffer size for shuffling the dataset

# Augmentation parameters
AUGMENTATION = True     # Use data augmentation
JITTER_PADDING = 30     # (px) Expand image size by this amount for random jitter and later crop
USE_ELASTIC_DEFORMATION = False  # Use elastic deformation in augmentation

# Training parameters
GEN_LEARNING_RATE = 2e-4    # Learning rate for generator
GEN_BETA_1 = 0.5            # Beta 1 parameter for Adam optimizer (generator)
GEN_BETA_2 = 0.999          # Beta 2 parameter for Adam optimizer (generator)

DISC_LEARNING_RATE = 2e-4   # Learning rate for discriminator
DISC_BETA_1 = 0.5           # Beta 1 parameter for Adam optimizer (discriminator)
DISC_BETA_2 = 0.999         # Beta 2 parameter for Adam optimizer (discriminator)

LAMBDA = [100.0,             # Lambda parameter for l1 loss
          0.0             # Lambda parameter for catch-only loss  
]

BATCH_SIZE = 1          # Batch size for training
EPOCHS = 100            # Number of training epochs

# Evaluation parameters-------------------------------------------------------------
EVALUATION_FREQUENCY = 5               # Evaluate metrics every N epochs
SAVE_EVAL_IMAGES_FREQUENCY = 100       # Save evaluation images every N epochs
N_TRAIN_SAMPLES = 50                   # Number of training samples for evaluation

# Fish classification parameters
CATCH_THRESHOLD = 0.2                  # Above this = catch (practical threshold for fish detection)
# -----------------------------------------------------------------------------------
BASE_DIR = '/home/peder/fish-forecast/GAN-model'
CHECKPOINT_DIR = f'{BASE_DIR}/training_ckpts/{ITERATION_NAME}'
LOG_DIR = f'{BASE_DIR}/logs/{ITERATION_NAME}'
IMAGES_DIR = f'{BASE_DIR}/ckpt_images/{ITERATION_NAME}'
TEST_EVALUATION_DIR = f'{IMAGES_DIR}/test_evaluation_plots'
TRAIN_EVALUATION_DIR = f'{IMAGES_DIR}/train_evaluation_plots'
PREDICTION_DIR = f'{IMAGES_DIR}/prediction'
# -------------------------
os.makedirs(TEST_EVALUATION_DIR, exist_ok=True)
os.makedirs(TRAIN_EVALUATION_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
# -------------------------
# Single GPU setup
tf.keras.backend.set_floatx('float32')
logger.info('Running on single GPU')


# Prepare datasets
dirs = list(CHANNEL_PREFIXES.keys())
manager = DatasetManager(input_dirs=dirs,
                         channel_prefixes=CHANNEL_PREFIXES,
                         output_dir=OUTPUT_DIR,
                         output_prefix=OUTPUT_PREFIX,
                         output_channels=OUTPUT_CHANNELS,
                         img_height=IMG_HEIGHT,
                         img_width=IMG_WIDTH,
                         data_augmentation=AUGMENTATION,
                         jitter_padding=JITTER_PADDING,
                         use_elastic_deformation=USE_ELASTIC_DEFORMATION)

# Preconditioning: check if all images are present and of the same size
validation_passed = manager.check_all_image_shapes()
if not validation_passed:
    raise ValueError("Dataset validation failed")

train_ds, test_ds, validation_ds = manager.create_train_test_validation_datasets(
    validation_size=VALIDATION_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER
)

logger.info(f"Samples: train: {len(manager.get_train_dates())}, test: {len(manager.get_test_dates())}, validation: {len(manager.get_validation_dates())}")
# Build models
# Generator uses, training=True both in training and inference
generator = Generator(IMG_HEIGHT, 
                      IMG_WIDTH, 
                      INPUT_CHANNELS, 
                      OUTPUT_CHANNELS)
# Use the custom U-net with residual connections generator
# generator = Generator_minimal_residual(IMG_HEIGHT,
#                                        IMG_WIDTH,
#                                        INPUT_CHANNELS,
#                                        OUTPUT_CHANNELS)
tf.keras.utils.plot_model(generator, 
                          show_shapes=True, 
                          to_file='Generator.png',
                          rankdir='TB') # have worked with LR
discriminator = Discriminator(IMG_HEIGHT, 
                              IMG_WIDTH, 
                              INPUT_CHANNELS, 
                              OUTPUT_CHANNELS)
tf.keras.utils.plot_model(discriminator, 
                          show_shapes=True, 
                          to_file='Discriminator.png',
                          rankdir='TB')
# Optimizers
gen_opt = tf.keras.optimizers.Adam(GEN_LEARNING_RATE, beta_1=GEN_BETA_1, beta_2=GEN_BETA_2)
disc_opt = tf.keras.optimizers.Adam(DISC_LEARNING_RATE, beta_1=DISC_BETA_1, beta_2=DISC_BETA_2)

# Checkpointing
ckpt_pref = os.path.join(CHECKPOINT_DIR, 'ckpt')
# Track the current epoch for proper restoration
current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                 discriminator_optimizer=disc_opt,
                                 generator=generator,
                                 discriminator=discriminator,
                                 epoch=current_epoch)

# Initialize checkpoint directory and evaluation
start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(
    f"{LOG_DIR}/{start_timestamp}/fit")

# Setup evaluation
evaluator = ImageEvaluator(threshold=CATCH_THRESHOLD)
training_evaluator = TrainingEvaluator(
    evaluator=evaluator,
    log_dir=f"{LOG_DIR}/{start_timestamp}",
    eval_frequency=EVALUATION_FREQUENCY,
    save_images_frequency=SAVE_EVAL_IMAGES_FREQUENCY
)

# Train step with tf.function optimization
@tf.function
def train_step(inputs):
    inp, tar = inputs
    with tf.GradientTape(persistent=True) as tape:
        gen_out = generator(inp, training=True)
        d_real = discriminator([inp, tar], training=True)
        d_fake = discriminator([inp, gen_out], training=True)

        g_loss_dict = generator_loss(d_fake, 
                                     gen_out, 
                                     tar, 
                                     LAMBDA)
        d_loss_dict = discriminator_loss(d_real, d_fake)

    # Extract the losses we want to use
    g_loss = g_loss_dict["g_total_loss"]
    d_loss = d_loss_dict["d_loss"]

    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    d_grads = tape.gradient(d_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss_dict, d_loss_dict

# Epoch-based fit loop with optimized training
def fit(train_ds, test_ds, epochs, train_samples_count, start_epoch=0):
    start_time = time.time()
    
    # Calculate steps per epoch using actual training data (excluding test samples)
    # Use ceil division to account for the last incomplete batch
    steps_per_epoch = math.ceil(train_samples_count / BATCH_SIZE)
    logger.info(f"Training for {epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Starting from epoch {start_epoch + 1}")
    
    global_step = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Create progress bar for this epoch
        epoch_bar = tqdm(train_ds, 
                        desc=f"Epoch {epoch + 1}/{epochs}",
                        unit="batch",
                        leave=True)
        
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        batch_count = 0
        
        for batch, inputs in enumerate(epoch_bar):
            # Train step
            g_loss_dict, d_loss_dict = train_step(inputs)
            
            # Extract losses for display
            g_loss = g_loss_dict["g_total_loss"]
            d_loss = d_loss_dict["d_loss"]
            
            # Accumulate losses for epoch average
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            batch_count += 1
            
            # Log losses to tensorboard every 10 steps
            if global_step % 10 == 0:
                losses = g_loss_dict
                losses.update(d_loss_dict)
                with summary_writer.as_default():
                    for name, value in losses.items():
                        tf.summary.scalar(f'GAN-related-loss/{name}', value, step=global_step)
            
            # Update progress bar every batch
            epoch_bar.set_postfix({
                "gen_loss": f"{g_loss:.3f}", 
                "disc_loss": f"{d_loss:.3f}"
            })
            
            global_step += 1
        
        # Calculate epoch averages
        avg_gen_loss = epoch_gen_loss / batch_count
        avg_disc_loss = epoch_disc_loss / batch_count
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - "
                   f"avg_gen_loss: {avg_gen_loss:.4f}, avg_disc_loss: {avg_disc_loss:.4f}")
        
        # Periodic evaluation
        if training_evaluator.should_evaluate(epoch + 1):
            logger.info(f"Running evaluation at epoch {epoch + 1}")
            
            # Evaluate on ENTIRE test set instead of just 1 image
            all_test_targets = []
            all_test_predictions = []
            
            for test_batch in test_ds:  # Iterate through ALL test batches
                test_inp, test_tar = test_batch

                test_predictions = generator(test_inp, training=True)
                
                all_test_targets.append(test_tar)
                all_test_predictions.append(test_predictions)
            
            # Concatenate all test results
            full_test_targets = tf.concat(all_test_targets, axis=0)
            full_test_predictions = tf.concat(all_test_predictions, axis=0)
            
            # Evaluate test set with threshold visualizations
            test_metrics = training_evaluator.evaluate_and_visualize_thresholds(
                full_test_targets, full_test_predictions, epoch + 1, 'test',
                images_save_dir=TEST_EVALUATION_DIR
            )
            
            logger.info(f"Evaluated on {full_test_targets.shape[0]} test images")
            
            # Print both quick and detailed summaries
            training_evaluator.print_quick_summary(test_metrics, epoch + 1, 'test')
            training_evaluator.print_metrics_summary(test_metrics, epoch + 1, 'test')
            training_evaluator.log_metrics_to_tensorboard(test_metrics, epoch + 1, summary_writer)

            # Use N_TRAIN_SAMPLES images from the training set
            train_samples = []
            train_targets = []
            for i, (inp, tar) in enumerate(train_ds.take(N_TRAIN_SAMPLES)):  
                train_samples.append(inp)
                train_targets.append(tar)

            train_inp_batch = tf.concat(train_samples, axis=0)
            train_tar_batch = tf.concat(train_targets, axis=0)
            train_predictions = generator(train_inp_batch, training=True)

            # Evaluate training set with threshold visualizations  
            train_metrics = training_evaluator.evaluate_and_visualize_thresholds(
                train_tar_batch, train_predictions, epoch + 1, 'train',
                images_save_dir=TRAIN_EVALUATION_DIR
            )
            
            logger.info(f"Evaluated on {train_tar_batch.shape[0]} train images")
            
            # Print both quick and detailed summaries
            training_evaluator.print_quick_summary(train_metrics, epoch + 1, 'train')
            training_evaluator.print_metrics_summary(train_metrics, epoch + 1, 'train')
            training_evaluator.log_metrics_to_tensorboard(train_metrics, epoch + 1, summary_writer)

        # Show predictions after each epoch using test dataset (preserves val set integrity)
        show_predictions(generator,
                       test_ds,
                       number=5,
                       save_path=PREDICTION_DIR,
                       steps=epoch + 1)  # Use epoch number instead of global step

        # Update current epoch and save checkpoint at end of each epoch
        if epoch % 25 == 0 and epoch > 0:
            current_epoch.assign(epoch + 1)  # Save the next epoch to start from
            checkpoint.save(file_prefix=ckpt_pref)
            logger.info(f"Checkpoint saved after epoch {epoch + 1}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Generate training curves
    logger.info("Generating training curves")
    training_evaluator.plot_training_curves(save_path=f'{IMAGES_DIR}/training_curves.png')
    
    logger.info("Training complete.")

if __name__ == '__main__':
    start_epoch = 0
    # Load existing checkpoint if available
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        logger.info(f"Restoring from checkpoint: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
        start_epoch = int(current_epoch.numpy())
        logger.info(f"Resuming training from epoch {start_epoch + 1}")
    else:
        logger.info("No checkpoint found, starting fresh training")

    # Save hyperparameters at the start of training
    # Save all hyperparameters for reproducibility
    hyperparams = collect_hyperparameters(locals())
    save_hyperparameters(hyperparams, IMAGES_DIR)
    
    # # Optionally, update EPOCHS to train for additional epochs
    # EPOCHS += 250
    fit(train_ds,
        test_ds,       
        EPOCHS,
        len(manager.get_train_dates()),
        start_epoch=start_epoch)

    # Generate final images on the test set and save them
    # This provides final unbiased evaluation on the test set
    if not tf.io.gfile.exists(f"{IMAGES_DIR}/final_test"):
        tf.io.gfile.makedirs(f"{IMAGES_DIR}/final_test")

    # Create final evaluation directory
    os.makedirs(f"{IMAGES_DIR}/final_test_evaluation", exist_ok=True)

    logger.info("Generating final test predictions and performing evaluation")
    
    # Collect all test predictions and targets for evaluation
    all_test_targets = []
    all_test_predictions = []
    
    # Generate images and collect data for evaluation
    for idx, (inp, tar) in enumerate(test_ds.take(len(manager.get_test_dates()))):
        gen_out = generator(inp, training=True)  # Use training=True for GANs as per TF tutorial
        
        # Save comparison images
        if len(tar.shape) == 4:
            tar_for_image = tf.squeeze(tar, axis=0)
        else:
            tar_for_image = tar
            
        if len(gen_out.shape) == 4:
            gen_out_for_image = tf.squeeze(gen_out, axis=0)
        else:
            gen_out_for_image = gen_out

        comparison = tf.concat([tar_for_image, gen_out_for_image], axis=1)  # shape: (256, 512, 3)
        img = comparison.numpy()
        # Map images back to [0, 255] from [-1,1]
        img = ((img + 1) * 127.5).astype('uint8')
        plt.imsave(f"{IMAGES_DIR}/final_test/{idx}.png", img)
        
        # Collect for evaluation (keep original dimensions)
        all_test_targets.append(tar)
        all_test_predictions.append(gen_out)
    
    # Concatenate all test results
    full_test_targets = tf.concat(all_test_targets, axis=0)
    full_test_predictions = tf.concat(all_test_predictions, axis=0)
    
    # Perform comprehensive final evaluation
    logger.info(f"Performing final evaluation on {full_test_targets.shape[0]} test images")
    
    # Final comprehensive evaluation with all visualizations
    final_test_metrics = training_evaluator.evaluate_and_visualize_thresholds(
        full_test_targets, full_test_predictions, epoch=EPOCHS, dataset_type='final_test',
        images_save_dir=f"{IMAGES_DIR}/final_test_evaluation"
    )
    
    # Print comprehensive final results to terminal
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION RESULTS")
    print("="*80)
    training_evaluator.print_metrics_summary(final_test_metrics, EPOCHS, 'final_test')
    
    # Print key metrics summary
    print(f"\nKEY FINAL RESULTS:")
    print(f"   Test Pixel Accuracy:    {final_test_metrics.get('pixel_accuracy', 0):.4f}")
    print(f"   Test Catch F1:          {final_test_metrics.get('catch_f1', 0):.4f}")
    print(f"   Test Catch IoU:         {final_test_metrics.get('iou_catch', 0):.4f}")
    print(f"   Test Catch Precision:   {final_test_metrics.get('catch_precision', 0):.4f}")
    print(f"   Test Catch Recall:      {final_test_metrics.get('catch_recall', 0):.4f}")
    print(f"   Test Mean IoU:          {final_test_metrics.get('mean_iou', 0):.4f}")
    print(f"   Test L1 Loss:           {final_test_metrics.get('l1_loss', 0):.4f}")
    print(f"   Test SSIM:              {final_test_metrics.get('ssim', 0):.4f}")
    print(f"   Test PSNR:              {final_test_metrics.get('psnr', 0):.4f}")
    print("="*80)
    
    # # Log to TensorBoard
    # training_evaluator.log_metrics_to_tensorboard(final_test_metrics, EPOCHS, 'final_test')
    
    # Save final metrics to JSON file
    import json
    final_metrics_path = f"{IMAGES_DIR}/final_test_metrics.json"
    with open(final_metrics_path, 'w') as f:
        # Convert numpy values to regular Python types for JSON serialization
        json_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in final_test_metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    # Save final metrics to text file for easy reading
    final_metrics_txt_path = f"{IMAGES_DIR}/final_test_metrics.txt"
    with open(final_metrics_txt_path, 'w') as f:
        f.write("FINAL TEST SET EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {ITERATION_NAME}\n")
        f.write(f"Evaluation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Images: {full_test_targets.shape[0]}\n")
        f.write(f"Catch Threshold: {CATCH_THRESHOLD}\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Pixel Accuracy:    {final_test_metrics.get('pixel_accuracy', 0):.4f}\n")
        f.write(f"Catch F1:          {final_test_metrics.get('catch_f1', 0):.4f}\n")
        f.write(f"Catch IoU:         {final_test_metrics.get('iou_catch', 0):.4f}\n")
        f.write(f"Catch Precision:   {final_test_metrics.get('catch_precision', 0):.4f}\n")
        f.write(f"Catch Recall:      {final_test_metrics.get('catch_recall', 0):.4f}\n")
        f.write(f"Mean IoU:          {final_test_metrics.get('mean_iou', 0):.4f}\n")
        f.write(f"L1 Loss:           {final_test_metrics.get('l1_loss', 0):.4f}\n")
        f.write(f"SSIM:              {final_test_metrics.get('ssim', 0):.4f}\n")
        f.write(f"PSNR:              {final_test_metrics.get('psnr', 0):.4f}\n\n")
        
        f.write("ALL METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric_name, value in sorted(final_test_metrics.items()):
            f.write(f"{metric_name:25s}: {value:.4f}\n")
    
    # Generate final comparison plots
    training_evaluator.evaluator.plot_comparison(
        full_test_targets, full_test_predictions, 
        num_samples=min(5, full_test_targets.shape[0]),
        save_path=f"{IMAGES_DIR}/final_test_evaluation/final_comparison.png"
    )
    
    # Generate final confusion matrix
    training_evaluator.evaluator.plot_confusion_matrix(
        full_test_targets, full_test_predictions,
        save_path=f"{IMAGES_DIR}/final_test_evaluation/final_confusion_matrix.png"
    )
    
    # Generate final metrics summary plot
    training_evaluator.evaluator.plot_metrics_summary(
        final_test_metrics,
        save_path=f"{IMAGES_DIR}/final_test_evaluation/final_metrics_summary.png"
    )
    
    logger.info(f"Final test evaluation completed!")
    logger.info(f"Results saved to:")
    logger.info(f"  - JSON: {final_metrics_path}")
    logger.info(f"  - Text: {final_metrics_txt_path}")
    logger.info(f"  - Images: {IMAGES_DIR}/final_test/")
    logger.info(f"  - Evaluation plots: {IMAGES_DIR}/final_test_evaluation/")
    
    print(f"\nFILES SAVED:")
    print(f"   Test images: {IMAGES_DIR}/final_test/")
    print(f"   Metrics (JSON): {final_metrics_path}")
    print(f"   Metrics (Text): {final_metrics_txt_path}")
    print(f"   Evaluation plots: {IMAGES_DIR}/final_test_evaluation/")