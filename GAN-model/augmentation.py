import tensorflow as tf
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from utils.logger import logger


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image, crop_height=128, crop_width=512):
    # get dynamic shape
    shape = tf.shape(input_image)
    H, W = shape[0], shape[1]
    # how many pixels we can slide over
    max_off_h = H - crop_height + 1
    max_off_w = W - crop_width  + 1

    # pick one random offset for both images
    offset_h = tf.random.uniform((), 0, max_off_h, dtype=tf.int32)
    offset_w = tf.random.uniform((), 0, max_off_w, dtype=tf.int32)

    # slice out the same window
    inp_crop = input_image[
        offset_h:offset_h + crop_height,
        offset_w:offset_w + crop_width,
        :
    ]
    tar_crop = real_image[
        offset_h:offset_h + crop_height,
        offset_w:offset_w + crop_width,
        :
    ]

    return inp_crop, tar_crop



@tf.function()
def random_jitter(input_image, real_image, padding: int = 100, height: int = 256, width: int = 256):
    input_image, real_image = resize(input_image, 
                                     real_image, 
                                     height + padding, 
                                     width + padding)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image, height, width )

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def elastic_deformation_tf(input_image, real_image, grid_size=4, sigma=10):
    """
    Apply elastic deformation to both input and real images using the same deformation field.
    
    Args:
        input_image: Input tensor image
        real_image: Target tensor image  
        grid_size: Size of control grid for deformation
        sigma: Standard deviation of random displacements
    
    Returns:
        Tuple of deformed (input_image, real_image)
    """
    def elastic_deform_numpy(input_np, real_np, grid_size, sigma):
        H, W = input_np.shape[:2]
        
        # Control grid coordinates
        xs = np.linspace(0, W-1, grid_size)
        ys = np.linspace(0, H-1, grid_size)
        
        # Generate ONE set of random displacements for both images
        dx = np.random.randn(grid_size, grid_size) * sigma
        dy = np.random.randn(grid_size, grid_size) * sigma
        
        # Spline interpolation
        k = min(3, grid_size-1)
        interp_dx = RectBivariateSpline(ys, xs, dy, kx=k, ky=k)
        interp_dy = RectBivariateSpline(ys, xs, dx, kx=k, ky=k)
        
        # Build full-resolution warp (same for both images)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = X + interp_dy(np.arange(H), np.arange(W))
        map_y = Y + interp_dx(np.arange(H), np.arange(W))
        
        # Apply same warp to both images
        def warp_image(image_np):
            if len(image_np.shape) == 3:
                warped = np.zeros_like(image_np)
                for c in range(image_np.shape[2]):
                    warped[:, :, c] = map_coordinates(image_np[:, :, c], 
                                                    [map_y.ravel(), map_x.ravel()],
                                                    order=1).reshape(image_np.shape[:2])
            else:
                warped = map_coordinates(image_np, [map_y.ravel(), map_x.ravel()],
                                       order=1).reshape(image_np.shape)
            return warped.astype(np.float32)
        
        input_warped = warp_image(input_np)
        real_warped = warp_image(real_np)
        
        return input_warped, real_warped
    
    # Convert to numpy, apply same deformation to both images, convert back
    input_np = input_image.numpy()
    real_np = real_image.numpy()
    
    # Apply same deformation field to both images
    input_deformed, real_deformed = elastic_deform_numpy(input_np, real_np, grid_size, sigma)
    
    return tf.constant(input_deformed), tf.constant(real_deformed)

def apply_elastic_deformation(input_image, real_image, grid_size=4, sigma=10):
    """
    TensorFlow wrapper for elastic deformation.
    """
    input_deformed, real_deformed = tf.py_function(
        func=lambda inp, real: elastic_deformation_tf(inp, real, grid_size, sigma),
        inp=[input_image, real_image],
        Tout=[tf.float32, tf.float32]
    )
    
    # Preserve shape information
    input_deformed.set_shape(input_image.shape)
    real_deformed.set_shape(real_image.shape)
    
    return input_deformed, real_deformed

def augment(input_image, real_image, 
           use_jitter=True, 
           use_elastic=False,
           jitter_padding=100,
           jitter_height=256, 
           jitter_width=256,
           elastic_grid_size=3,
           elastic_sigma=10):
    """
    Main augmentation function that applies selected augmentation methods.
    
    Args:
        input_image: Input tensor image
        real_image: Target tensor image
        use_jitter: Whether to apply random jitter (resize + crop + flip)
        use_elastic: Whether to apply elastic deformation
        jitter_padding: Padding for jitter augmentation
        jitter_height: Target height after jitter
        jitter_width: Target width after jitter  
        elastic_grid_size: Grid size for elastic deformation
        elastic_sigma: Sigma parameter for elastic deformation
    
    Returns:
        Tuple of augmented (input_image, real_image)
    """
    # Apply elastic deformation first (if enabled) since it works on original resolution
    # logger.debug(f'before any Augmentation \ninp: {tf.shape(input_image)} \ntar: {tf.shape(real_image)}')
    if use_elastic:
        input_image, real_image = apply_elastic_deformation(
            input_image, real_image, elastic_grid_size, elastic_sigma
        )

    # logger.debug(f'After elastic \ninp: {tf.shape(input_image)} \ntar: {tf.shape(real_image)}')
    
    # Apply jitter augmentation (if enabled)
    if use_jitter:
        input_image, real_image = random_jitter(
            input_image, real_image, jitter_padding, jitter_height, jitter_width
        )
    
    # logger.debug(f'After jitter \ninp: {tf.shape(input_image)} \ntar: {tf.shape(real_image)}')
    return input_image, real_image


if __name__ == "__main__":
    pass