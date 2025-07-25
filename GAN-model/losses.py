import tensorflow as tf

# Adversarial Loss
def adversarial_loss_logits(disc_output, target_is_real: bool):
    """
    Adversarial loss with binary cross entropy.
    Args:
		disc_output: discriminator logits.
		target_is_real: True for real targets, False for fake.
    Returns:
      	Scalar BCE loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    labels = tf.ones_like(disc_output) if target_is_real else tf.zeros_like(disc_output)
    return bce(labels, disc_output)

# Generator Loss
def generator_loss(disc_generated_output, gen_output, target, lambda_list) -> dict:
    """
    Compute generator loss: adversarial + L1.
    Args:
        disc_generated_output: discriminator logits on generated images.
        gen_output: generated images.
        target: ground truth images.
        lambda_list: list of weights for losses.
    Returns:
        dict with total loss and individual components.
    """
    gan_loss = adversarial_loss_logits(disc_generated_output, True)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Losses from FA
    gamma = 2.0
    alpha = 1.0
    pcl     = pixel_constraint_loss(target, gen_output)
    pscl    = pixel_sum_constraint_loss(target, gen_output)
    fll     = focal_l1_loss(target, gen_output, gamma, alpha)
    mfll    = masked_focal_l1_loss(target, gen_output, gamma, alpha)
    acd     = abs_catch_diff(target, gen_output) 
    lco     = l1_catch_only(target, gen_output)
    mc      = mask_catch(target, gen_output)

    # Scale losses by their respective weights
    l1_loss *= lambda_list[0]  # Scale L1 loss
    lco *= lambda_list[1]       # Scale pixel constraint loss

    # Combine losses with weights
    total_loss = gan_loss + l1_loss + lco

    # Pack losses in a dict for easy logging during training
    loss_dict =  {
        'g_total_loss': total_loss,
        'gan_loss': gan_loss,
        'l1_loss': l1_loss,
        'pixel_constraint_loss': pcl,
        'pixel_sum_constraint_loss': pscl,
        'focal_l1_loss': fll,
        'masked_focal_l1_loss': mfll,
        'masked_catches': mc,
        'abs_catch_diff': acd,
        'l1_catch_only': lco
    }
    
    return loss_dict


# Discriminator Loss
def discriminator_loss(disc_real_output, disc_generated_output) -> dict:
    """
    Compute discriminator loss: real vs. fake.
    Args:
		disc_real_output: logits for real images.
		disc_generated_output: logits for generated images.
    Returns:
      	total discriminator loss.
    """
    real_loss = adversarial_loss_logits(disc_real_output, True)
    fake_loss = adversarial_loss_logits(disc_generated_output, False)
    losses = {
        'd_real_loss': real_loss, 
        'd_fake_loss': fake_loss,
        'd_loss': real_loss + fake_loss
    } 
    return losses

# Pixel Constraints
def pixel_constraint_loss(y_true, y_pred) -> tf.Tensor:
    """
    Penalize L1 difference in non-zero target regions.
    """
    mask = tf.cast(y_true > 0.0, tf.float32)
    return tf.reduce_sum(mask * tf.abs(y_true - y_pred))


def pixel_sum_constraint_loss(y_true, y_pred) -> tf.Tensor:
    """
    Penalize difference in max pixel-sum between true and generated images.
    """
    true_sum = tf.reduce_sum(y_true, axis=[1,2])
    gen_sum = tf.reduce_sum(y_pred, axis=[1,2])
    return tf.abs(tf.reduce_max(true_sum) - tf.reduce_max(gen_sum))

# Focal L1 Losses
def focal_l1_loss(
    y_true, y_pred,
    gamma: float,
    alpha: float
) -> tf.Tensor:
    """
    Focal L1: |error|^gamma scaled by alpha.
    """
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(alpha * tf.pow(error, gamma))

def masked_focal_l1_loss(
    y_true, y_pred,
    gamma: float,
    alpha: float
) -> tf.Tensor:
    """
    Apply focal L1 only on positive target regions.
    """
    mask = tf.cast(y_true > 0.0, tf.float32)
    error = tf.abs(y_true - y_pred)
    weighted = alpha * tf.pow(error, gamma)
    return tf.reduce_sum(weighted * mask) / (tf.reduce_sum(mask) + 1e-8)


# the model now goes into some state where it predicts the same thing regardless of the input, i want to give a score of how well the model is at predicting catch (at all[any red points at the map (>200, <100, <100)]). At the same time i dont want to limit prediction to only use as much read as in the target image

# Attempt at a loss function that penalizes the model for not predicting any red points in the generated image, but at the same time allows the model to use as much red as it wants
def mask_catch(y_true, y_pred) -> tf.Tensor:
    """
    Loss function that penalizes the model for not predicting any red points in the generated image.
    """
    # Define a mask for red points in the target image
    mask = tf.cast(y_true > 0.0, tf.float32)
    
    # Calculate the mean of the predicted image
    pred_mean = tf.reduce_mean(y_pred)
    
    # Calculate the mean of the target image
    true_mean = tf.reduce_mean(y_true)
    
    # Calculate the custom loss
    loss = tf.reduce_mean(mask * (pred_mean - true_mean))
    
    return loss


def abs_catch_diff(target, generated):
    """
    Isolate marked catch locations in the target and generated images. Sum the red pixels in both images and compute the absolute difference.
    """
    target_red =  target[:, :, 0] - target[:, :, 1]
    generated_red = generated[:, :, 0] - generated[:, :, 1]

    target_red = tf.where(target_red > 0.5, 1.0, 0.0)
    generated_red = tf.where(generated_red > 0.5, 1.0, 0.0)

    sum_target = tf.reduce_sum(target_red)
    sum_generated = tf.reduce_sum(generated_red)
    diff = tf.abs(sum_target - sum_generated)
    return diff


def l1_catch_only(target, generated):
    """
    Isolate catch locations (white pixels) in grayscale target and generated images.
    Compare only the catch regions between target and generated.
    """
    # For grayscale images, catches are white pixels (high values)
    # Threshold to identify catch locations (white pixels)
    catch_threshold = 0.2  # Adjust this threshold as needed
    
    # Create binary masks for catch locations
    target_catches = tf.cast(target > catch_threshold, tf.float32)
    generated_catches = tf.cast(generated > catch_threshold, tf.float32)
    
    # Calculate L1 loss only on catch regions
    diff = tf.reduce_mean(tf.abs(target_catches - generated_catches))
    return diff


def new_l1_catch_only(target, generated):
    """
    Calculates the L1 loss ONLY on the pixels that are considered "catch"
    in the ground truth target image. Stable version.
    """
    catch_threshold = 0.2
    
    catch_mask = tf.cast(target > catch_threshold, tf.float32)
    num_catch_pixels = tf.reduce_sum(catch_mask)
    
    # Early return if no catch pixels exist
    if tf.equal(num_catch_pixels, 0):
        return tf.constant(0.0)
    
    error = tf.abs(target - generated)
    masked_error = error * catch_mask
    
    # Normalize by the number of catch pixels (more stable)
    loss = tf.reduce_sum(masked_error) / num_catch_pixels
    
    return loss

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(15, 20))  # Increased figure size for 5 rows
    
    for i in range(1, 6):
        # Load images using matplotlib
        target = plt.imread(f"/home/peder/fish-forecast/GAN-model/test_images/target_{i}.png")
        predicted = plt.imread(f"/home/peder/fish-forecast/GAN-model/test_images/prediction_{i}.png")

        # Convert to TensorFlow tensors
        target = tf.constant(target, dtype=tf.float32)
        predicted = tf.constant(predicted, dtype=tf.float32)

        # Normalize to [-1, 1] range (assuming input is [0, 1])
        target = target * 2.0 - 1.0
        predicted = predicted * 2.0 - 1.0

        catch_threshold = 0.7  # Adjust this threshold as needed
    
        # Create binary masks for catch locations
        target_catches = tf.cast(target > catch_threshold, tf.float32)
        generated_catches = tf.cast(predicted > catch_threshold, tf.float32)
        
        # Calculate L1 loss only on catch regions
        diff1 = tf.reduce_mean(tf.abs(target_catches - generated_catches))


        # Calculate L2 loss only on catch regions
        diff2 = tf.reduce_mean(tf.square(target_catches - generated_catches))
        print(f"Img pair{i}: L1: {diff1.numpy():.4f}, L2: {diff2.numpy():.4f}")

        plt.subplot(5, 3, 1)
        plt.title(f"Target {i}")
        plt.imshow(target_catches, cmap="gray")
        plt.axis('off')
        
        plt.subplot(5, 3, 2)
        plt.title(f"Predicted {i}")
        plt.imshow(generated_catches, cmap="gray")
        plt.axis('off')
        
        plt.subplot(5, 3, 3)
        plt.title(f"Difference {i}\nL1: {diff1.numpy():.4f}\nL2: {diff2.numpy():.4f}")
        diff_img = tf.abs(target_catches - generated_catches)
        im = plt.imshow(diff_img, cmap="gray")
        plt.axis('off')
        plt.colorbar(im, shrink=0.6)

    plt.tight_layout()
    plt.savefig("/home/peder/fish-forecast/GAN-model/loss_output.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved visualization to loss_output.png")
