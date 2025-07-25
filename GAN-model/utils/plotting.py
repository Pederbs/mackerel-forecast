
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def generate_unraveled_image(model: tf.keras.Model,
                    test_input: tf.Tensor,
                    target: tf.Tensor,
                    channel_names: dict[str, str],
                    cols: int = 5,
                    save_path: str = './ckpt_images',
                    steps: int = None) -> None:
    """
    Given a trained `model`, a batch `test_input`, and its `target`,
    shows:
     - each input channel
     - the ground-truth target
     - the model's prediction
    """
    # Run the model in inference mode
    prediction = model(test_input, training=False)

    # How many input channels?
    num_input_channels = test_input.shape[-1]
    # Total panels = inputs + ground truth + prediction
    total_panels = num_input_channels + 2

    # Determine grid size
    if cols < 3:
        cols = min(5, total_panels)
    rows = (total_panels + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2, rows * 1))

    # 1) Plot each input channel
    for i in range(num_input_channels):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow((test_input[0,...,i] + 1)/2, cmap='gray')
        ax.set_title(f"{list(channel_names.values())[i]}")
        ax.axis('off')

    # 2) Ground truth
    ax = fig.add_subplot(rows, cols, num_input_channels + 1)
    ax.imshow((target[0] + 1)/2)
    ax.set_title("Ground Truth")
    ax.axis('off')

    # 3) Model prediction
    ax = fig.add_subplot(rows, cols, num_input_channels + 2)
    ax.imshow((prediction[0] + 1)/2)
    ax.set_title("Prediction")
    ax.axis('off')


    # Create save path if it doesn't exist
    if not tf.io.gfile.exists(save_path):
        tf.io.gfile.makedirs(save_path)

    plt.tight_layout()
    if steps is not None:
        plt.savefig(f'{save_path}/unraveled_prediction_{steps}.png', dpi=300)
    else:
        plt.savefig(f'{save_path}/unraveled_prediction.png', dpi=300)


def show_predictions(model: tf.keras.Model,
                    dataset: tf.data.Dataset,
                    number: int = 1,
                    save_path: str = './ckpt_images',
                    steps: int = None) -> None:
    """
    Show predictions for multiple images from the dataset, with titles only on top.
    Args:
        model: The trained model
        dataset: Dataset containing input-target pairs
        number: Number of image pairs to show
        save_path: Path to save the generated image
        steps: Current training step (optional)
    """
    # Ensure that the folder exists
    os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(figsize=(4, number * 2))
    # Shuffle the dataset and get iterator for random samples
    dataset_iter = iter(dataset.shuffle(buffer_size=1000).take(number))

    for img in range(number):
        inp, tar = next(dataset_iter)
        # print(f"==========Input shape: {inp.shape}", flush=True)
        # Handle both batched and unbatched datasets
        if inp.shape.rank == 3:
            inp = tf.expand_dims(inp, axis=0)
        if tar.shape.rank == 3:
            tar = tf.expand_dims(tar, axis=0)
            
        prediction = model(inp, training=True)

        # Plot target - handle both batched and unbatched
        ax1 = fig.add_subplot(number, 2, 2*img + 1)
        if img == 0:
            ax1.set_title("Target Image")
        # If tar has batch dimension, use tar[0], otherwise use tar directly
        target_img = tar[0] if tar.shape.rank == 4 else tar
        ax1.imshow((target_img * 0.5) + 0.5)
        ax1.axis('off')
        
        # Plot prediction
        ax2 = fig.add_subplot(number, 2, 2*img + 2)
        if img == 0:
            ax2.set_title("Predicted Image")
        ax2.imshow((prediction[0] * 0.5) + 0.5)
        ax2.axis('off')

    plt.tight_layout()
    if steps is not None:
        plt.savefig(f'{save_path}/prediction_{steps}.png', dpi=300)
    else:
        plt.savefig(f'{save_path}/prediction.png', dpi=300)