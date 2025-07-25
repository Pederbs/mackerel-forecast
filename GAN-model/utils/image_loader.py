import tensorflow as tf
import os
from typing import Union

class ImageLoader:
    def __init__(self, img_height: int, img_width: int):
        """
        Initialize the ImageLoader.

        Args:
            img_height (int): Desired image height after resizing.
            img_width (int): Desired image width after resizing.
        """
        self.img_height = img_height
        self.img_width = img_width

    def _load_and_preprocess(self, path: Union[str, os.PathLike], channels: int) -> tf.Tensor:
        """
        Load an image from disk, decode as PNG, resize, and normalize.

        Args:
            path (str or os.PathLike): Path to the image file.
            channels (int): Number of channels to decode (1 for grayscale, 3 for RGB).

        Returns:
            tf.Tensor: The preprocessed image tensor, normalized to [-1, 1].
        """
        # Read and decode the PNG file with the specified number of channels
        file_content = tf.io.read_file(str(path))
        image = tf.io.decode_png(file_content, channels=channels)
        
        # Convert to float and resize
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        
        # Normalize to [-1, 1] range
        image = image / 127.5 - 1
        
        return image

    def load_input_image(self, date: str, dir_path: Union[str, os.PathLike], prefix: str) -> tf.Tensor:
        """
        Load and preprocess a single-channel (grayscale) input image for a given date.

        Args:
            date (str): The date or identifier for the image (used in filename).
            dir_path (str or os.PathLike): Directory containing the image.
            prefix (str): Prefix for the image filename.

        Returns:
            tf.Tensor: The preprocessed single-channel image tensor.

        Raises:
            FileNotFoundError: If the input image file does not exist.
        """
        file_name = f"{prefix}{date}.png"
        full_path = os.path.join(dir_path, file_name)
        return self._load_and_preprocess(full_path, channels=1)

    def load_target_image(self, date: str, dir_path: Union[str, os.PathLike], prefix: str, ch=1) -> tf.Tensor:
        """
        Load and preprocess a three-channel (RGB) target image for a given date.

        Args:
            date (str): The date or identifier for the image (used in filename).
            dir_path (str or os.PathLike): Directory containing the image.
            prefix (str): Prefix for the image filename.

        Returns:
            tf.Tensor: The preprocessed three-channel image tensor.

        Raises:
            FileNotFoundError: If the target image file does not exist.
        """
        file_name = f"{prefix}{date}.png"
        full_path = os.path.join(dir_path, file_name)
        return self._load_and_preprocess(full_path, channels=ch)