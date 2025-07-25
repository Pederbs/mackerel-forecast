from typing import List, Dict, Tuple, Set
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.file_extractor import get_available_dates
from utils.image_loader import ImageLoader
from utils.logger import logger
from augmentation import random_jitter, augment

class DatasetManager:
    def __init__(
        self,
        input_dirs: List[str],
        channel_prefixes: Dict[str, str],
        output_dir: str,
        output_prefix: str = "catch_",
        output_channels=1,
        img_height: int = 128,
        img_width: int = 512,
        data_augmentation: bool = True,
        jitter_padding: int = 100,
        use_elastic_deformation: bool = False
    ):
        self.input_dirs = input_dirs
        self.channel_prefixes = channel_prefixes
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.output_channels = output_channels
        self.loader = ImageLoader(img_height, img_width)
        # Will hold latest split lists
        self._train_dates: List[str] = []
        self._test_dates: List[str] = []
        self._validation_dates: List[str] = []
        # Member variables to control data augmentation
        self.apply_jitter = data_augmentation
        self.padding = jitter_padding
        self.use_elastic_deformation = use_elastic_deformation

    def get_common_and_missing_dates(self) -> Tuple[Set[str], Set[str]]:
        """
        Returns:
          - common_dates: dates present in all input_dirs and in output_dir
          - missing_dates: dates present in all input_dirs but missing in output_dir
        """
        date_sets = [get_available_dates(d) for d in self.input_dirs]
        common_input = set.intersection(*date_sets) if date_sets else set()
        target_dates = get_available_dates(self.output_dir)
        common = common_input & target_dates
        missing = common_input.difference(target_dates)
        return common, missing

    def split_dates(
        self,
        validation_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Split dates into train/validation lists without building datasets.

        Args:
            validation_size: Fraction for validation split.
            random_state: Seed for reproducibility.
        Returns:
            A tuple (train_dates, validation_dates).
        """
        common_dates, _ = self.get_common_and_missing_dates()
        # Print info about dates in dataset and abort if none exists
        if len(common_dates) == 0:
            logger.error(f'Found: {len(common_dates)} common dates in the dataset, aborting run!')
            exit()
        else:
            logger.info(f'Found: {len(common_dates)} common dates in the dataset')

        train_dates, validation_dates = train_test_split(
            sorted(common_dates), test_size=validation_size, random_state=random_state
        )
        self._train_dates = train_dates
        self._validation_dates = validation_dates
        # Clear test dates since we're not using them
        self._test_dates = []
        return train_dates, validation_dates

    def get_train_dates(self) -> List[str]:
        """
        Return the most recently generated training date list.
        """
        return self._train_dates

    def get_test_dates(self) -> List[str]:
        """
        Return the most recently generated testing date list.
        """
        return self._test_dates

    def get_validation_dates(self) -> List[str]:
        """
        Return the most recently generated validation date list.
        """
        return self._validation_dates

    def create_train_validation_datasets(
        self,
        validation_size: float = 0.2,
        random_state: int = 42,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create train and validation datasets.
        """
        # 1) generate and store date splits
        self._train_dates, self._validation_dates = self.split_dates(validation_size, random_state)
        if not self._train_dates or not self._validation_dates:
            logger.warning("Train or validation date list is empty; check data availability.")

        # 2) build the tf.data pipelines
        self.apply_jitter = True
        train_ds = (
            tf.data.Dataset.from_tensor_slices(self._train_dates)
            .shuffle(shuffle_buffer, seed=random_state)
            .map(self.load_example, num_parallel_calls=num_parallel_calls)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        self.apply_jitter = False
        validation_ds = (
            tf.data.Dataset.from_tensor_slices(self._validation_dates)
            .map(self.load_example, num_parallel_calls=num_parallel_calls)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        return train_ds, validation_ds

    def create_train_test_validation_datasets(
        self,
        validation_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42,
        batch_size: int = 32,
        shuffle_buffer: int = 1000,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train, test, and validation datasets with proper separation.
        Test and validation sizes are percentages of the total dataset.
        """
        # Get all common dates
        common_dates, _ = self.get_common_and_missing_dates()
        if len(common_dates) == 0:
            logger.error(f'Found: {len(common_dates)} common dates in the dataset, aborting run!')
            exit()
        else:
            logger.info(f'Found: {len(common_dates)} common dates in the dataset')

        # Calculate actual numbers for clarity
        total_samples = len(common_dates)
        num_validation = int(total_samples * validation_size)
        num_test = int(total_samples * test_size)
        num_train = total_samples - num_validation - num_test
        
        logger.info(f"Dataset split: train={num_train}, test={num_test}, validation={num_validation}")
        
        # Split all dates directly into three sets
        sorted_dates = sorted(common_dates)
        np.random.seed(random_state)
        shuffled_dates = np.random.permutation(sorted_dates)
        
        validation_dates = shuffled_dates[:num_validation].tolist()
        test_dates = shuffled_dates[num_validation:num_validation + num_test].tolist()
        train_dates = shuffled_dates[num_validation + num_test:].tolist()
        
        # Store the splits
        self._train_dates = train_dates
        self._test_dates = test_dates 
        self._validation_dates = validation_dates
        
        if not self._train_dates or not self._test_dates or not self._validation_dates:
            logger.warning("Train, test, or validation date list is empty; check data availability.")

        # Build the tf.data pipelines
        self.apply_jitter = True
        train_ds = (
            tf.data.Dataset.from_tensor_slices(self._train_dates)
            .shuffle(shuffle_buffer, seed=random_state)
            .map(self.load_example, num_parallel_calls=num_parallel_calls)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        
        self.apply_jitter = False
        test_ds = (
            tf.data.Dataset.from_tensor_slices(self._test_dates)
            .map(self.load_example, num_parallel_calls=num_parallel_calls)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        
        validation_ds = (
            tf.data.Dataset.from_tensor_slices(self._validation_dates)
            .map(self.load_example, num_parallel_calls=num_parallel_calls)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        
        return train_ds, test_ds, validation_ds

    def load_sample_for_date(self, date: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns (input_image, target_image) for the given date.
        """
        channel_imgs = [
            self.loader.load_input_image(date, d, self.channel_prefixes[d])
            for d in self.input_dirs
        ]
        input_image = tf.concat(channel_imgs, axis=-1)
        target_image = self.loader.load_target_image(
            date, self.output_dir, self.output_prefix, ch=self.output_channels
        )
        return input_image, target_image

    def load_example(self, date: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Wraps `load_sample_for_date` for use in tf.data pipelines (accepts a scalar string tensor).
        """
        def _py_load(d):
            # d may be a tf.Tensor in graph, so get the bytes via d.numpy()
            byte_str = d.numpy() if hasattr(d, "numpy") else d
            date_str = byte_str.decode("utf-8")
            return self.load_sample_for_date(date_str)

        inp, tar = tf.py_function(
            func=_py_load,
            inp=[date],
            Tout=[tf.float32, tf.float32]
        )
        # fix shapes
        num_ch = len(self.input_dirs)
        inp.set_shape([self.loader.img_height, self.loader.img_width, num_ch])
        tar.set_shape([self.loader.img_height, self.loader.img_width, self.output_channels])

        if self.apply_jitter:
            # Apply augmentation with optional elastic deformation
            aug_inp, aug_tar = augment(
                inp, tar, 
                use_jitter=True,
                use_elastic=self.use_elastic_deformation,
                jitter_padding=self.padding, 
                jitter_height=self.loader.img_height, 
                jitter_width=self.loader.img_width
            )
            return aug_inp, aug_tar
        else:
            return inp, tar

    def plot_example(
        self,
        date: str,
        cols: int = None
    ) -> None:
        """
        Plot all input channels and target image for one date.
        """
        inp, tgt = self.load_sample_for_date(date)
        num_ch = inp.shape[-1]
        total = num_ch + 1
        if cols is None:
            cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))
        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(num_ch):
            ax = plt.subplot(rows, cols, i + 1)
            plt.imshow((inp[..., i] + 1) / 2, cmap='gray')
            ax.set_title(f"Channel {i}")
            ax.axis('off')
        ax = plt.subplot(rows, cols, total)
        plt.imshow((tgt + 1) / 2)
        ax.set_title("Target")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def check_all_image_shapes(self) -> bool:
        """
        Verifies that all common dates have images matching the expected shape.
        Returns True if all OK, False otherwise.
        """
        common, missing = self.get_common_and_missing_dates()
        if missing:
            logger.warning(f"Missing dates: {sorted(missing)}")
        mismatches = []
        for date in sorted(common):
            inp, tgt = self.load_sample_for_date(date)
            if (
                inp.shape[0] != self.loader.img_height
                or inp.shape[1] != self.loader.img_width
                or tgt.shape[0] != self.loader.img_height
                or tgt.shape[1] != self.loader.img_width
            ):
                mismatches.append(date)
                logger.error(f"Shape mismatch {date}: inp {inp.shape}, tgt {tgt.shape}")
        if mismatches:
            logger.error(f"Total mismatches: {len(mismatches)}")
            return False
        logger.info(f"All {len(common)} samples OK with shape ({self.loader.img_height},{self.loader.img_width})")
        return True
