"""
Utility functions for saving and managing hyperparameters.
"""
import os
import json
import datetime
import tensorflow as tf
from utils.logger import logger


def save_hyperparameters(hyperparams_dict, save_dir):
    """
    Save all hyperparameters to files in the specified directory.
    
    Args:
        hyperparams_dict (dict): Dictionary containing all hyperparameters
        save_dir (str): Directory path where hyperparameters will be saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add metadata to the hyperparameters
    hyperparams_dict.update({
        'TIMESTAMP': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'TENSORFLOW_VERSION': tf.__version__,
    })
    
    # Save as both JSON and human-readable text
    
    # Save as JSON for programmatic access
    json_path = os.path.join(save_dir, '00_hyperparameters.json')
    with open(json_path, 'w') as f:
        json.dump(hyperparams_dict, f, indent=2, default=str)
    
    # # Save as human-readable text (appears first in directory listing)
    # txt_path = os.path.join(save_dir, '00_hyperparameters.txt')
    # with open(txt_path, 'w') as f:
    #     f.write("=" * 60 + "\n")
    #     f.write("HYPERPARAMETERS FOR TRAINING RUN\n")
    #     f.write("=" * 60 + "\n\n")
        
    #     # Group parameters by category
    #     categories = {
    #         'Dataset Parameters': ['DATASET_PATH', 'CHANNEL_PREFIXES', 'OUTPUT_DIR', 'OUTPUT_PREFIX'],
    #         'Image Parameters': ['IMG_HEIGHT', 'IMG_WIDTH', 'INPUT_CHANNELS', 'OUTPUT_CHANNELS'],
    #         'Dataset Split Parameters': ['VALIDATION_SIZE', 'TEST_SIZE', 'RANDOM_STATE', 'SHUFFLE_BUFFER'],
    #         'Augmentation Parameters': ['AUGMENTATION', 'JITTER_PADDING', 'USE_ELASTIC_DEFORMATION'],
    #         'Evaluation Parameters': ['EVALUATION_FREQUENCY', 'SAVE_EVAL_IMAGES_FREQUENCY', 'EVALUATION_THRESHOLD'],
    #         'Generator Training Parameters': ['GEN_LEARNING_RATE', 'GEN_BETA_1', 'GEN_BETA_2'],
    #         'Discriminator Training Parameters': ['DISC_LEARNING_RATE', 'DISC_BETA_1', 'DISC_BETA_2'],
    #         'Loss Parameters': ['LAMBDA'],
    #         'Training Parameters': ['BATCH_SIZE', 'EPOCHS'],
    #         'Directory Parameters': ['ITERATION_NAME', 'BASE_DIR', 'CHECKPOINT_DIR', 'LOG_DIR', 'IMAGES_DIR'],
    #         'Metadata': ['TIMESTAMP', 'TENSORFLOW_VERSION']
    #     }
        
    #     for category, param_names in categories.items():
    #         f.write(f"{category}:\n")
    #         f.write("-" * len(category) + "\n")
    #         for param in param_names:
    #             if param in hyperparams_dict:
    #                 value = hyperparams_dict[param]
    #                 if isinstance(value, dict):
    #                     f.write(f"  {param}:\n")
    #                     for k, v in value.items():
    #                         f.write(f"    {k}: {v}\n")
    #                 elif isinstance(value, list):
    #                     f.write(f"  {param}: {value}\n")
    #                 else:
    #                     f.write(f"  {param}: {value}\n")
    #         f.write("\n")
    
    logger.info(f"Hyperparameters saved to {json_path}")


def collect_hyperparameters(locals_dict):
    """
    Collect hyperparameters from the training script's local variables.
    
    Args:
        locals_dict (dict): Local variables dictionary from the training script
        
    Returns:
        dict: Dictionary containing all hyperparameters
    """
    # Define the hyperparameters we want to collect
    hyperparam_names = [
        # Dataset parameters
        'DATASET_PATH', 'CHANNEL_PREFIXES', 'OUTPUT_DIR', 'OUTPUT_PREFIX',
        
        # Image parameters
        'IMG_HEIGHT', 'IMG_WIDTH', 'INPUT_CHANNELS', 'OUTPUT_CHANNELS',
        
        # Dataset parameters
        'VALIDATION_SIZE', 'TEST_SIZE', 'RANDOM_STATE', 'SHUFFLE_BUFFER',
        
        # Augmentation parameters
        'AUGMENTATION', 'JITTER_PADDING', 'USE_ELASTIC_DEFORMATION',
        
        # Evaluation parameters
        'EVALUATION_FREQUENCY', 'SAVE_EVAL_IMAGES_FREQUENCY', 'EVALUATION_THRESHOLD',
        
        # Training parameters
        'GEN_LEARNING_RATE', 'GEN_BETA_1', 'GEN_BETA_2',
        'DISC_LEARNING_RATE', 'DISC_BETA_1', 'DISC_BETA_2',
        'LAMBDA', 'BATCH_SIZE', 'EPOCHS',
        
        # Checkpointing and logging
        'ITERATION_NAME', 'BASE_DIR', 'CHECKPOINT_DIR', 'LOG_DIR', 'IMAGES_DIR',
    ]
    
    # Collect hyperparameters that exist in locals
    hyperparams = {}
    for name in hyperparam_names:
        if name in locals_dict:
            hyperparams[name] = locals_dict[name]
    
    return hyperparams
