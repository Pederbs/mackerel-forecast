import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.logger import logger

class ImageEvaluator:
    """
    Comprehensive evaluation metrics for fish catch prediction.
    Handles 3-class classification: land (-1), sea (0), catch (1).
    Uses practical thresholds for real-world fish forecasting.
    """
    
    def __init__(self, threshold: float = 0.2, land_threshold: float = -0.3):
        """
        Initialize evaluator with practical thresholds.
        
        Args:
            threshold: Catch threshold - predictions above this are classified as catch
            land_threshold: Land threshold - predictions below this are classified as land
        """
        self.catch_threshold = threshold
        self.land_threshold = land_threshold
    
    def normalize_to_01(self, images: tf.Tensor) -> tf.Tensor:
        """Normalize images from [-1, 1] to [0, 1] range for visualization."""
        return (images + 1.0) / 2.0
    
    def classify_predictions(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Classify predictions into 3 classes: land (-1), sea (0), catch (1).
        
        Args:
            y_pred: Predicted values in [-1, 1] range
            
        Returns:
            Classified predictions with values {-1, 0, 1}
        """
        # Start with sea (0) as default
        classified = tf.zeros_like(y_pred)
        
        # Land: predictions below land_threshold
        land_mask = y_pred < self.land_threshold
        classified = tf.where(land_mask, -1.0, classified)
        
        # Catch: predictions above catch_threshold
        catch_mask = y_pred > self.catch_threshold
        classified = tf.where(catch_mask, 1.0, classified)
        
        return classified
    
    # =================== PIXEL-LEVEL METRICS ===================
    
    def pixel_accuracy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Pixel accuracy for 3-class classification.
        Percentage of pixels that are correctly classified.
        """
        y_pred_classified = self.classify_predictions(y_pred)
        correct_pixels = tf.cast(tf.equal(y_true, y_pred_classified), tf.float32)
        return tf.reduce_mean(correct_pixels)
    
    def mean_pixel_accuracy(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Mean pixel accuracy - average accuracy per class.
        Computes accuracy for each class separately then averages.
        """
        y_pred_classified = self.classify_predictions(y_pred)
        
        # Accuracy for each class
        class_accuracies = []
        for class_val in [-1.0, 0.0, 1.0]:  # land, sea, catch
            class_mask = tf.cast(y_true == class_val, tf.float32)
            class_correct = tf.reduce_sum(class_mask * tf.cast(y_pred_classified == class_val, tf.float32))
            class_total = tf.reduce_sum(class_mask) + 1e-8
            class_acc = class_correct / class_total
            class_accuracies.append(class_acc)
        
        return tf.reduce_mean(class_accuracies)
    
    # =================== IoU METRICS ===================
    
    def iou_score(self, y_true: tf.Tensor, y_pred: tf.Tensor, class_id: int = 1) -> tf.Tensor:
        """
        Intersection over Union (IoU) for specific class.
        
        Args:
            y_true: Ground truth with values {-1, 0, 1}
            y_pred: Predictions in [-1, 1] range
            class_id: Class to compute IoU for (-1=land, 0=sea, 1=catch)
        """
        y_pred_classified = self.classify_predictions(y_pred)
        
        # Create binary masks for the specified class
        y_true_class = tf.cast(y_true == class_id, tf.float32)
        y_pred_class = tf.cast(y_pred_classified == class_id, tf.float32)
        
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        
        return intersection / (union + 1e-8)
    
    def mean_iou(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Mean IoU across all classes.
        """
        iou_land = self.iou_score(y_true, y_pred, class_id=-1)
        iou_sea = self.iou_score(y_true, y_pred, class_id=0)
        iou_catch = self.iou_score(y_true, y_pred, class_id=1)
        
    
    def mean_iou(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Mean IoU across all classes.
        """
        iou_land = self.iou_score(y_true, y_pred, class_id=-1)
        iou_sea = self.iou_score(y_true, y_pred, class_id=0)
        iou_catch = self.iou_score(y_true, y_pred, class_id=1)
        
        return (iou_land + iou_sea + iou_catch) / 3.0
    
    # =================== PRECISION, RECALL, F1 METRICS ===================
    
    def precision_recall_f1(self, y_true: tf.Tensor, y_pred: tf.Tensor, class_id: int = 1) -> Dict[str, tf.Tensor]:
        """
        Precision, Recall, F1-score for specific class.
        
        Args:
            y_true: Ground truth with values {-1, 0, 1}
            y_pred: Predictions in [-1, 1] range
            class_id: Class to compute metrics for (-1=land, 0=sea, 1=catch)
        """
        y_pred_classified = self.classify_predictions(y_pred)
        
        # Create binary masks for the specified class
        y_true_class = tf.cast(y_true == class_id, tf.float32)
        y_pred_class = tf.cast(y_pred_classified == class_id, tf.float32)
        
        # True/False Positives/Negatives
        tp = tf.reduce_sum(y_true_class * y_pred_class)
        fp = tf.reduce_sum((1 - y_true_class) * y_pred_class)
        fn = tf.reduce_sum(y_true_class * (1 - y_pred_class))
        
        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # =================== INTENSITY METRICS ===================
    
    def l1_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """L1 loss - used in Pix2Pix paper."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def l2_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """L2 loss (MSE) - standard regression metric."""
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def ssim_score(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Structural Similarity Index (SSIM).
        Used in image quality assessment.
        """
        y_true_norm = self.normalize_to_01(y_true)
        y_pred_norm = self.normalize_to_01(y_pred)
        return tf.image.ssim(y_true_norm, y_pred_norm, max_val=1.0)
    
    def psnr_score(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Peak Signal-to-Noise Ratio (PSNR).
        Higher values indicate better quality.
        """
        y_true_norm = self.normalize_to_01(y_true)
        y_pred_norm = self.normalize_to_01(y_pred)
        return tf.image.psnr(y_true_norm, y_pred_norm, max_val=1.0)
    
    # =================== CATCH-SPECIFIC METRICS ===================
    
    def catch_detection_metrics(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Comprehensive catch detection metrics for practical fish forecasting.
        Focuses on catch class (1) performance.
        """
        return self.precision_recall_f1(y_true, y_pred, class_id=1)
    
    def catch_area_error(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Error in total catch area prediction.
        Compares predicted vs actual catch area.
        """
        y_pred_classified = self.classify_predictions(y_pred)
        
        true_catch_area = tf.reduce_sum(tf.cast(y_true == 1, tf.float32))
        pred_catch_area = tf.reduce_sum(tf.cast(y_pred_classified == 1, tf.float32))
        
        return tf.abs(true_catch_area - pred_catch_area) / (true_catch_area + 1e-8)
    
    def l1_catch_only(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        L1 loss computed only on catch regions.
        """
        # Create binary masks for catch locations
        catch_mask = tf.cast(y_true == 1, tf.float32)
        
        # Calculate L1 loss only on catch regions
        diff = tf.abs(y_true - y_pred) * catch_mask
        total_catch_pixels = tf.reduce_sum(catch_mask) + 1e-8
        
        return tf.reduce_sum(diff) / total_catch_pixels
    
    # =================== COMPREHENSIVE EVALUATION ===================
    
    def evaluate_batch(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, float]:
        """
        Comprehensive evaluation of a batch of images using 3-class classification.
        
        Args:
            y_true: Ground truth images [batch, height, width, channels] with values {-1, 0, 1}
            y_pred: Predicted images [batch, height, width, channels] in [-1, 1] range
            
        Returns:
            Dictionary of all evaluation metrics
        """
        metrics = {}
        
        # Convert to float32 if needed
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Pixel-level metrics
        metrics['pixel_accuracy'] = self.pixel_accuracy(y_true, y_pred).numpy()
        metrics['mean_pixel_accuracy'] = self.mean_pixel_accuracy(y_true, y_pred).numpy()
        
        # IoU metrics for all classes
        metrics['iou_land'] = self.iou_score(y_true, y_pred, class_id=-1).numpy()
        metrics['iou_sea'] = self.iou_score(y_true, y_pred, class_id=0).numpy()
        metrics['iou_catch'] = self.iou_score(y_true, y_pred, class_id=1).numpy()
        metrics['mean_iou'] = self.mean_iou(y_true, y_pred).numpy()
        
        # Precision/Recall/F1 for all classes
        for class_id, class_name in [(-1, 'land'), (0, 'sea'), (1, 'catch')]:
            class_metrics = self.precision_recall_f1(y_true, y_pred, class_id=class_id)
            for metric_name, value in class_metrics.items():
                metrics[f'{class_name}_{metric_name}'] = value.numpy()
        
        # Intensity metrics (Pix2Pix standard)
        metrics['l1_loss'] = self.l1_loss(y_true, y_pred).numpy()
        metrics['l2_loss'] = self.l2_loss(y_true, y_pred).numpy()
        metrics['ssim'] = tf.reduce_mean(self.ssim_score(y_true, y_pred)).numpy()
        metrics['psnr'] = tf.reduce_mean(self.psnr_score(y_true, y_pred)).numpy()
        
        # Catch-specific metrics
        metrics['catch_area_error'] = self.catch_area_error(y_true, y_pred).numpy()
        metrics['l1_catch_only'] = self.l1_catch_only(y_true, y_pred).numpy()
        
        return metrics
    
    # =================== VISUALIZATION ===================
    
    def plot_comparison(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                       num_samples: int = 3, save_path: str = None):
        """
        Plot side-by-side comparison of ground truth and predictions.
        """
        # Ensure tensors have a batch dimension for consistent processing
        if len(y_true.shape) == 3:
            y_true = tf.expand_dims(y_true, axis=0)
            y_pred = tf.expand_dims(y_pred, axis=0)

        # Limit num_samples to the actual number of images in the batch
        actual_batch_size = y_true.shape[0]
        num_samples = min(num_samples, actual_batch_size)

        y_true_norm = self.normalize_to_01(y_true)
        y_pred_norm = self.normalize_to_01(y_pred)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Ground truth
            # --- Fix: Use index `i` instead of `0` ---
            axes[i, 0].imshow(y_true_norm[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Ground Truth {i+1}')
            axes[i, 0].axis('off')
            
            # Prediction
            # --- Fix: Use index `i` instead of `0` ---
            axes[i, 1].imshow(y_pred_norm[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Prediction {i+1}')
            axes[i, 1].axis('off')
            
            # Difference
            diff = tf.abs(y_true_norm[i] - y_pred_norm[i])
            im = axes[i, 2].imshow(diff[:, :, 0], cmap='hot', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Absolute Difference {i+1}')
            axes[i, 2].axis('off')
            
            # Add colorbar for difference
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            # --- Fix: Close the plot to free memory ---
            plt.close(fig)
        else:
            plt.show()
    
    def plot_catch_overlay(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                          num_samples: int = 3, save_path: str = None):
        """
        Plot overlayed catch predictions on ground truth.
        """
        y_true_norm = self.normalize_to_01(y_true)
        y_pred_norm = self.normalize_to_01(y_pred)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # Create RGB overlay: GT in red, predictions in green
            overlay = np.zeros((y_true.shape[1], y_true.shape[2], 3))
            
            # Ground truth catch areas in red
            gt_catch = y_true_norm[i, :, :, 0] > 0.5
            overlay[:, :, 0] = gt_catch
            
            # Predicted catch areas in green
            pred_catch = y_pred_norm[i, :, :, 0] > 0.5
            overlay[:, :, 1] = pred_catch
            
            # Overlap will appear yellow (red + green)
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Catch Overlay {i+1}\n(GT=Red, Pred=Green, Overlap=Yellow)')
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                             save_path: str = None):
        """
        Plot confusion matrix for 3-class classification (land, sea, catch).
        """
        y_pred_classified = self.classify_predictions(y_pred)
        
        y_true_flat = y_true.numpy().flatten()
        y_pred_flat = y_pred_classified.numpy().flatten()
        
        # Convert to integer labels for confusion matrix
        # -1 -> 0, 0 -> 1, 1 -> 2
        y_true_int = ((y_true_flat + 1) * 0.5 + 0.5).astype(int)  # -1->0, 0->1, 1->2
        y_pred_int = ((y_pred_flat + 1) * 0.5 + 0.5).astype(int)
        
        cm = confusion_matrix(y_true_int, y_pred_int)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Land', 'Sea', 'Catch'],
                   yticklabels=['Land', 'Sea', 'Catch'])
        plt.title(f'Confusion Matrix - Fish Catch Prediction (threshold={self.catch_threshold})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_summary(self, metrics: Dict[str, float], save_path: str = None):
        """
        Plot a summary of all evaluation metrics.
        """
        # Group metrics by category
        pixel_metrics = {k: v for k, v in metrics.items() if 'pixel' in k}
        iou_metrics = {k: v for k, v in metrics.items() if 'iou' in k}
        catch_metrics = {k: v for k, v in metrics.items() if 'catch_' in k and k != 'catch_area_error'}
        loss_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['l1', 'l2', 'ssim', 'psnr'])}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pixel metrics
        if pixel_metrics:
            axes[0, 0].bar(pixel_metrics.keys(), pixel_metrics.values())
            axes[0, 0].set_title('Pixel-Level Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # IoU metrics
        if iou_metrics:
            axes[0, 1].bar(iou_metrics.keys(), iou_metrics.values())
            axes[0, 1].set_title('IoU Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Catch detection metrics
        if catch_metrics:
            catch_viz = {k.replace('catch_', ''): v for k, v in catch_metrics.items() 
                        if k not in ['catch_true_positives', 'catch_false_positives', 
                                   'catch_false_negatives', 'catch_true_negatives']}
            axes[1, 0].bar(catch_viz.keys(), catch_viz.values())
            axes[1, 0].set_title('Catch Detection Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Loss metrics
        if loss_metrics:
            axes[1, 1].bar(loss_metrics.keys(), loss_metrics.values())
            axes[1, 1].set_title('Loss/Quality Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    # =================== THRESHOLD VISUALIZATION ===================
    
    def plot_threshold_illustration(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                                   thresholds: List[float] = [-0.5, 0.0, 0.5],
                                   save_path: str = None):
        """
        Visualize how different thresholds affect 3-class classification.
        
        Args:
            y_true: Ground truth images [-1, 1] range
            y_pred: Predicted images [-1, 1] range  
            thresholds: List of threshold values to test
            save_path: Path to save the plot
        """
        import matplotlib.patches as patches
        
        # Take first image from batch
        if len(y_true.shape) == 4:
            y_true = y_true[0]
            y_pred = y_pred[0]
        
        y_true_norm = self.normalize_to_01(y_true)
        y_pred_norm = self.normalize_to_01(y_pred)
        
        fig, axes = plt.subplots(3, len(thresholds) + 2, figsize=(20, 12))
        
        # Column 0: Ground Truth
        axes[0, 0].imshow(y_true_norm[:, :, 0], cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('Ground Truth\n(Continuous)')
        axes[0, 0].axis('off')
        
        # Ground truth classification (3-class)
        gt_land = y_true[:, :, 0] < -0.3  # Land (very negative)
        gt_sea = (y_true[:, :, 0] >= -0.3) & (y_true[:, :, 0] < 0.3)  # Sea (around 0)
        gt_catch = y_true[:, :, 0] >= 0.3  # Catch (positive)
        
        gt_classified = tf.zeros_like(y_true[:, :, 0])
        gt_classified = tf.where(gt_land, 0.0, gt_classified)    # Land = 0 (dark)
        gt_classified = tf.where(gt_sea, 0.5, gt_classified)     # Sea = 0.5 (gray)
        gt_classified = tf.where(gt_catch, 1.0, gt_classified)   # Catch = 1 (bright)
        
        axes[1, 0].imshow(gt_classified, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth\n(3-Class)')
        axes[1, 0].axis('off')
        
        # Legend
        legend_elements = [
            patches.Patch(color='black', label='Land'),
            patches.Patch(color='gray', label='Sea'),
            patches.Patch(color='yellow', label='Catch')
        ]
        axes[2, 0].legend(handles=legend_elements, loc='center')
        axes[2, 0].axis('off')
        
        # Column 1: Prediction
        axes[0, 1].imshow(y_pred_norm[:, :, 0], cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title('Prediction\n(Continuous)')
        axes[0, 1].axis('off')
        
        axes[1, 1].text(0.5, 0.5, 'Thresholds →', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=16, rotation=90)
        axes[1, 1].axis('off')
        axes[2, 1].axis('off')
        
        # Columns 2+: Different thresholds
        for i, threshold in enumerate(thresholds, 2):
            # Apply threshold to prediction
            pred_classified = tf.zeros_like(y_pred[:, :, 0])
            
            # Simple binary threshold for catch vs non-catch
            pred_catch = y_pred[:, :, 0] > threshold
            pred_classified = tf.where(pred_catch, 1.0, pred_classified)
            pred_classified = tf.where(~pred_catch, 0.5, pred_classified)  # Assume non-catch is sea
            
            axes[0, i].imshow(pred_classified, cmap='viridis', vmin=0, vmax=1)
            axes[0, i].set_title(f'Threshold = {threshold:.1f}')
            axes[0, i].axis('off')
            
            # Calculate metrics for this threshold
            temp_evaluator = ImageEvaluator(threshold=threshold)
            metrics = temp_evaluator.evaluate_batch(
                tf.expand_dims(y_true, 0), 
                tf.expand_dims(y_pred, 0)
            )
            
            # Show key metrics
            axes[1, i].text(0.5, 0.8, f"Precision: {metrics['catch_precision']:.3f}", 
                           ha='center', transform=axes[1, i].transAxes)
            axes[1, i].text(0.5, 0.6, f"Recall: {metrics['catch_recall']:.3f}", 
                           ha='center', transform=axes[1, i].transAxes)
            axes[1, i].text(0.5, 0.4, f"F1: {metrics['catch_f1']:.3f}", 
                           ha='center', transform=axes[1, i].transAxes)
            axes[1, i].text(0.5, 0.2, f"IoU: {metrics['iou_catch']:.3f}", 
                           ha='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
            
            # Difference map
            diff = tf.abs(gt_classified - pred_classified)
            axes[2, i].imshow(diff, cmap='Reds', vmin=0, vmax=1)
            axes[2, i].set_title(f'Error Map')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_fish_detection_analysis(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                                    save_path: str = None):
        """
        Comprehensive fish detection analysis with practical metrics for forecasting.
        Updated to use the 3-class system with catch threshold = 0.2.
        """
        from sklearn.metrics import precision_recall_curve, auc
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        y_true_flat = y_true.numpy().flatten()
        y_pred_flat = y_pred.numpy().flatten()
        
        # Define class masks for ground truth (discrete values)
        y_true_land = (y_true_flat == -1).astype(int)
        y_true_sea = (y_true_flat == 0).astype(int)
        y_true_catch = (y_true_flat == 1).astype(int)
        
        # 1. Precision-Recall for Catch Detection (better for imbalanced data)
        precision, recall, pr_thresholds = precision_recall_curve(y_true_catch, y_pred_flat)
        pr_auc = auc(recall, precision)
        
        # Find optimal threshold for F1 score
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_pr_idx = np.argmax(f1_scores)
        optimal_pr_threshold = pr_thresholds[optimal_pr_idx] if len(pr_thresholds) > optimal_pr_idx else 0.0
        
        axes[0, 0].plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 0].scatter(recall[optimal_pr_idx], precision[optimal_pr_idx], 
                          color='red', s=100, label=f'Optimal F1 (t={optimal_pr_threshold:.2f})')
        axes[0, 0].set_xlabel('Recall (Sensitivity)')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision-Recall: Fish Catch Detection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Detection Rate vs False Alarm Rate (practical for forecasting)
        threshold_range = np.linspace(-1, 1, 100)
        detection_rates = []
        false_alarm_rates = []
        sea_false_alarms = []
        
        for thresh in threshold_range:
            pred_catch = (y_pred_flat > thresh).astype(int)
            
            # Detection rate (recall for catch)
            detection_rate = (pred_catch & y_true_catch).sum() / (y_true_catch.sum() + 1e-8)
            detection_rates.append(detection_rate)
            
            # False alarm rate (predicting catch in non-catch areas)
            non_catch = 1 - y_true_catch
            false_alarm_rate = (pred_catch & non_catch).sum() / (non_catch.sum() + 1e-8)
            false_alarm_rates.append(false_alarm_rate)
            
            # False alarms specifically in sea areas (more relevant than land)
            sea_false_alarm_rate = (pred_catch & y_true_sea).sum() / (y_true_sea.sum() + 1e-8)
            sea_false_alarms.append(sea_false_alarm_rate)
        
        axes[0, 1].plot(false_alarm_rates, detection_rates, 'g-', linewidth=2, label='Overall')
        axes[0, 1].plot(sea_false_alarms, detection_rates, 'b-', linewidth=2, label='Sea Only')
        axes[0, 1].set_xlabel('False Alarm Rate')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].set_title('Detection vs False Alarm Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Threshold vs Key Metrics
        f1_scores = []
        precisions = []
        recalls = []
        iou_scores = []
        
        for thresh in threshold_range:
            temp_evaluator = ImageEvaluator(threshold=thresh)
            metrics = temp_evaluator.evaluate_batch(
                tf.expand_dims(y_true, 0), 
                tf.expand_dims(y_pred, 0)
            )
            
            f1_scores.append(metrics['catch_f1'])
            precisions.append(metrics['catch_precision'])
            recalls.append(metrics['catch_recall'])
            iou_scores.append(metrics['iou_catch'])
        
        # Find optimal thresholds
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = threshold_range[optimal_f1_idx]
        
        optimal_iou_idx = np.argmax(iou_scores)
        optimal_iou_threshold = threshold_range[optimal_iou_idx]
        
        axes[1, 0].plot(threshold_range, f1_scores, 'r-', linewidth=2, label='F1 Score')
        axes[1, 0].plot(threshold_range, precisions, 'b-', linewidth=2, label='Precision')
        axes[1, 0].plot(threshold_range, recalls, 'g-', linewidth=2, label='Recall')
        axes[1, 0].plot(threshold_range, iou_scores, 'm-', linewidth=2, label='IoU')
        
        axes[1, 0].axvline(x=optimal_f1_threshold, color='red', linestyle='--', alpha=0.7,
                          label=f'Best F1: {optimal_f1_threshold:.2f}')
        axes[1, 0].axvline(x=self.catch_threshold, color='black', linestyle='--', alpha=0.7,
                          label=f'Current: {self.catch_threshold:.2f}')
        
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Threshold vs Performance Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Class Distribution and Prediction Quality
        # Ground truth distribution
        gt_land_count = y_true_land.sum()
        gt_sea_count = y_true_sea.sum()
        gt_catch_count = y_true_catch.sum()
        total_pixels = len(y_true_flat)
        
        # Predictions at current threshold
        pred_current = (y_pred_flat > self.catch_threshold).astype(int)
        pred_catch_count = pred_current.sum()
        pred_nocatch_count = (1 - pred_current).sum()
        
        # True positives, false positives, etc.
        tp = (pred_current & y_true_catch).sum()
        fp = (pred_current & (1 - y_true_catch)).sum()
        fn = ((1 - pred_current) & y_true_catch).sum()
        tn = ((1 - pred_current) & (1 - y_true_catch)).sum()
        
        # Create comparison bars
        categories = ['Ground Truth', 'Predictions', 'Errors']
        
        # Ground truth breakdown
        gt_values = [gt_land_count/total_pixels, gt_sea_count/total_pixels, gt_catch_count/total_pixels]
        pred_values = [0, pred_nocatch_count/total_pixels, pred_catch_count/total_pixels]
        error_values = [0, fp/total_pixels, fn/total_pixels]  # False positives and false negatives
        
        x = np.arange(len(categories))
        width = 0.25
        
        axes[1, 1].bar(x - width, [gt_land_count, 0, 0], width, label='Land', color='brown', alpha=0.7)
        axes[1, 1].bar(x - width, [0, gt_sea_count, 0], width, bottom=[gt_land_count, 0, 0], 
                      label='Sea', color='blue', alpha=0.7)
        axes[1, 1].bar(x - width, [0, 0, 0], width, 
                      bottom=[gt_land_count, gt_sea_count, 0], label='Catch (GT)', color='yellow', alpha=0.7)
        axes[1, 1].bar(x - width, [0, 0, gt_catch_count], width, 
                      bottom=[gt_land_count, gt_sea_count, 0], color='yellow', alpha=0.7)
        
        axes[1, 1].bar(x, [0, pred_nocatch_count, 0], width, label='No Catch (Pred)', color='lightblue', alpha=0.7)
        axes[1, 1].bar(x, [0, 0, pred_catch_count], width, label='Catch (Pred)', color='orange', alpha=0.7)
        
        axes[1, 1].bar(x + width, [0, 0, fp], width, label='False Pos', color='red', alpha=0.7)
        axes[1, 1].bar(x + width, [0, 0, fn], width, bottom=[0, 0, fp], label='False Neg', color='darkred', alpha=0.7)
        
        axes[1, 1].set_xlabel('Data Type')
        axes[1, 1].set_ylabel('Number of Pixels')
        axes[1, 1].set_title(f'Class Distribution & Errors (t={self.catch_threshold:.2f})')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Return practical recommendations
        recommendations = {
            'optimal_f1_threshold': optimal_f1_threshold,
            'optimal_iou_threshold': optimal_iou_threshold,
            'optimal_pr_threshold': optimal_pr_threshold,
            'current_f1': f1_scores[np.abs(threshold_range - self.catch_threshold).argmin()],
            'max_f1': max(f1_scores),
            'precision_recall_auc': pr_auc
        }
        
        return recommendations
    
    def plot_3class_confusion_matrix(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                                    land_threshold: float = -0.3,
                                    catch_threshold: float = 0.3,
                                    save_path: str = None):
        """
        Plot 3-class confusion matrix (land, sea, catch).
        """
        import numpy as np
        
        def classify_3class(values, land_thresh, catch_thresh):
            """Convert continuous values to 3-class labels."""
            labels = np.zeros_like(values, dtype=int)
            labels[values < land_thresh] = 0      # Land
            labels[values >= catch_thresh] = 2    # Catch  
            labels[(values >= land_thresh) & (values < catch_thresh)] = 1  # Sea
            return labels
        
        # Flatten and classify
        y_true_flat = y_true.numpy().flatten()
        y_pred_flat = y_pred.numpy().flatten()
        
        y_true_3class = classify_3class(y_true_flat, land_threshold, catch_threshold)
        y_pred_3class = classify_3class(y_pred_flat, land_threshold, catch_threshold)
        
        # Create confusion matrix
        import seaborn as sns
        cm = confusion_matrix(y_true_3class, y_pred_3class)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Land', 'Sea', 'Catch'],
                    yticklabels=['Land', 'Sea', 'Catch'])
        plt.title(f'3-Class Confusion Matrix\nLand < {land_threshold}, Catch ≥ {catch_threshold}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy info
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f}', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return cm


# =================== TRAINING INTEGRATION ===================

class TrainingEvaluator:
    """
    Integration class for evaluating metrics during GAN training.
    Logs to TensorBoard and displays metrics periodically.
    """
    
    def __init__(self, evaluator: ImageEvaluator, log_dir: str, 
                 eval_frequency: int = 100, save_images_frequency: int = 500):
        """
        Initialize training evaluator.
        
        Args:
            evaluator: ImageEvaluator instance
            log_dir: Directory for TensorBoard logs
            eval_frequency: Evaluate metrics every N epoch
            save_images_frequency: Save comparison images every N epoch
        """
        self.evaluator = evaluator
        self.eval_frequency = eval_frequency
        self.save_images_frequency = save_images_frequency
        
        # TensorBoard writers
        self.train_writer = tf.summary.create_file_writer(f"{log_dir}/train")
        self.test_writer = tf.summary.create_file_writer(f"{log_dir}/test")
        self.val_writer = tf.summary.create_file_writer(f"{log_dir}/validation")
        
        # Store metrics history - include all dataset types
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': [],
            'validation': []
        }
    
    def should_evaluate(self, epoch: int) -> bool:
        """Check if we should evaluate at this epoch."""
        return epoch % self.eval_frequency == 0
    
    def should_save_images(self, epoch: int) -> bool:
        """Check if we should save comparison images at this epoch."""
        return epoch % self.save_images_frequency == 0
    
    def log_metrics_to_tensorboard(self, metrics: Dict[str, float], epoch: int, 
                                 writer: tf.summary.SummaryWriter):
        """Log metrics to TensorBoard with proper categorization."""
        with writer.as_default():
            # 1. Overall Performance Metrics
            overall_metrics = {
                'pixel_accuracy': metrics.get('pixel_accuracy', 0),
                'mean_pixel_accuracy': metrics.get('mean_pixel_accuracy', 0),
                'mean_iou': metrics.get('mean_iou', 0)
            }
            for name, value in overall_metrics.items():
                tf.summary.scalar(f'overall/{name}', value, step=epoch)
            
            # 2. Class-Specific IoU
            iou_metrics = {
                'land': metrics.get('iou_land', 0),
                'sea': metrics.get('iou_sea', 0),
                'catch': metrics.get('iou_catch', 0)
            }
            for name, value in iou_metrics.items():
                tf.summary.scalar(f'iou/{name}', value, step=epoch)
            
            # 3. Precision, Recall, F1 for each class
            for class_name in ['land', 'sea', 'catch']:
                precision = metrics.get(f'{class_name}_precision', 0)
                recall = metrics.get(f'{class_name}_recall', 0)
                f1 = metrics.get(f'{class_name}_f1', 0)
                
                tf.summary.scalar(f'precision/{class_name}', precision, step=epoch)
                tf.summary.scalar(f'recall/{class_name}', recall, step=epoch)
                tf.summary.scalar(f'f1/{class_name}', f1, step=epoch)
            
            # 4. Catch-Specific Metrics (most important)
            catch_metrics = {
                'catch_precision': metrics.get('catch_precision', 0),
                'catch_recall': metrics.get('catch_recall', 0),
                'catch_f1': metrics.get('catch_f1', 0),
                'catch_area_error': metrics.get('catch_area_error', 0)
            }
            for name, value in catch_metrics.items():
                tf.summary.scalar(f'catch_detection/{name}', value, step=epoch)
            
            # 5. Image Quality Metrics
            quality_metrics = {
                'l1_loss': metrics.get('l1_loss', 0),
                'l2_loss': metrics.get('l2_loss', 0),
                'ssim': metrics.get('ssim', 0),
                'psnr': metrics.get('psnr', 0),
                'l1_catch_only': metrics.get('l1_catch_only', 0)
            }
            for name, value in quality_metrics.items():
                tf.summary.scalar(f'image_quality/{name}', value, step=epoch)
            
            # 6. Log all metrics under a general category as well
            for metric_name, value in metrics.items():
                tf.summary.scalar(f'all_metrics/{metric_name}', value, step=epoch)
            
            writer.flush()
    
    def evaluate_and_log(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                        epoch: int, dataset_type: str = 'train',
                        images_save_dir: str = None) -> Dict[str, float]:
        """
        Evaluate metrics and log to TensorBoard.
        
        Args:
            y_true: Ground truth batch
            y_pred: Predicted batch
            epoch: Training epoch
            dataset_type: 'train', 'test', 'val', or 'validation'
            images_save_dir: Directory to save comparison images
            
        Returns:
            Dictionary of metrics
        """
        # Evaluate metrics
        metrics = self.evaluator.evaluate_batch(y_true, y_pred)
        
        # Select appropriate writer
        if dataset_type == 'train':
            writer = self.train_writer
        elif dataset_type == 'test':
            writer = self.test_writer
        else:  # 'val' or 'validation'
            writer = self.val_writer
        
        # Log to TensorBoard
        self.log_metrics_to_tensorboard(metrics, epoch, writer)
        
        # Store in history (handle both 'val' and 'validation' keys)
        metrics_with_epoch = {'epoch': epoch, **metrics}
        if dataset_type in self.metrics_history:
            self.metrics_history[dataset_type].append(metrics_with_epoch)
        elif dataset_type == 'validation' and 'val' in self.metrics_history:
            self.metrics_history['val'].append(metrics_with_epoch)
        
        # Save comparison images if requested
        if self.should_save_images(epoch) and images_save_dir:
            save_path = f"{images_save_dir}/{dataset_type}_comparison_epoch_{epoch}.png"
            self.evaluator.plot_comparison(y_true, y_pred, num_samples=3, save_path=save_path)
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, float], epoch: int, dataset_type: str):
        """Print a comprehensive summary of all evaluation metrics."""
        print(f"\n{'='*60}")
        print(f"{dataset_type.upper()} EVALUATION - EPOCH {epoch}")
        print(f"{'='*60}")
        
        # 1. Overall Performance Metrics
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Pixel Accuracy      : {metrics.get('pixel_accuracy', 0):.4f}")
        print(f"  Mean Pixel Accuracy : {metrics.get('mean_pixel_accuracy', 0):.4f}")
        print(f"  Mean IoU            : {metrics.get('mean_iou', 0):.4f}")
        
        # 2. Class-Specific IoU
        print(f"\nCLASS-SPECIFIC IoU:")
        print(f"  Land IoU            : {metrics.get('iou_land', 0):.4f}")
        print(f"  Sea IoU             : {metrics.get('iou_sea', 0):.4f}")
        print(f"  Catch IoU           : {metrics.get('iou_catch', 0):.4f}")
        
        # 3. Precision, Recall, F1 for each class
        print(f"\nPRECISION, RECALL, F1:")
        for class_name in ['land', 'sea', 'catch']:
            precision = metrics.get(f'{class_name}_precision', 0)
            recall = metrics.get(f'{class_name}_recall', 0)
            f1 = metrics.get(f'{class_name}_f1', 0)
            print(f"  {class_name.capitalize():5s} - P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        
        # 4. Catch-Specific Metrics (most important for fish forecasting)
        print(f"\nFISH CATCH DETECTION:")
        print(f"  Catch Precision     : {metrics.get('catch_precision', 0):.4f}")
        print(f"  Catch Recall        : {metrics.get('catch_recall', 0):.4f}")
        print(f"  Catch F1 Score      : {metrics.get('catch_f1', 0):.4f}")
        print(f"  Catch Area Error    : {metrics.get('catch_area_error', 0):.4f}")
        
        # 5. Image Quality Metrics
        print(f"\nIMAGE QUALITY:")
        print(f"  L1 Loss             : {metrics.get('l1_loss', 0):.4f}")
        print(f"  L2 Loss             : {metrics.get('l2_loss', 0):.4f}")
        print(f"  SSIM                : {metrics.get('ssim', 0):.4f}")
        print(f"  PSNR                : {metrics.get('psnr', 0):.4f}")
        print(f"  L1 Catch Only       : {metrics.get('l1_catch_only', 0):.4f}")
        
        print(f"\n{'='*60}")
        
        # Log summary to logger as well
        logger.info(f"{dataset_type.upper()} Evaluation Summary:")
        logger.info(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f}")
        logger.info(f"  Catch F1: {metrics.get('catch_f1', 0):.4f}")
        logger.info(f"  Catch IoU: {metrics.get('iou_catch', 0):.4f}")
        logger.info(f"  L1 Loss: {metrics.get('l1_loss', 0):.4f}")
        logger.info(f"  SSIM: {metrics.get('ssim', 0):.4f}")
    
    def print_quick_summary(self, metrics: Dict[str, float], epoch: int, dataset_type: str):
        """Print a concise summary of the most important metrics during training."""
        catch_f1 = metrics.get('catch_f1', 0)
        pixel_acc = metrics.get('pixel_accuracy', 0)
        catch_iou = metrics.get('iou_catch', 0)
        l1_loss = metrics.get('l1_loss', 0)
        ssim = metrics.get('ssim', 0)
        
        print(f"\n{dataset_type.upper()} QUICK SUMMARY (Epoch {epoch}):")
        print(f"   Catch F1: {catch_f1:.4f} | Pixel Acc: {pixel_acc:.4f} | Catch IoU: {catch_iou:.4f}")
        print(f"   L1 Loss: {l1_loss:.4f} | SSIM: {ssim:.4f}")
        
        # Log the most important metrics
        logger.info(f"{dataset_type.upper()} - Epoch {epoch}: "
                   f"Catch F1={catch_f1:.4f}, Pixel Acc={pixel_acc:.4f}, "
                   f"Catch IoU={catch_iou:.4f}, L1={l1_loss:.4f}")

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves for key metrics."""
        if not self.metrics_history['train']:
            logger.error("No training metrics to plot.")
            return
        
        # Extract data
        train_data = self.metrics_history['train']
        val_data = self.metrics_history.get('val', [])
        
        train_epochs = [m['epoch'] for m in train_data]
        val_epochs = [m['epoch'] for m in val_data] if val_data else []
        
        # Key metrics to plot
        key_metrics = ['pixel_accuracy', 'mean_iou', 'l1_loss', 'ssim', 'catch_f1']
        available_metrics = [m for m in key_metrics if m in train_data[0]]
        
        if not available_metrics:
            logger.error("No key metrics available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:6]):  # Max 6 plots
            train_values = [m[metric] for m in train_data]
            
            axes[i].plot(train_epochs, train_values, 'b-', label='Train', alpha=0.7)
            
            if val_data:
                val_values = [m[metric] for m in val_data if metric in m]
                if val_values:
                    axes[i].plot(val_epochs, val_values, 'r-', label='Validation', alpha=0.7)
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Training epoch')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    # =================== THRESHOLD VISUALIZATION ===================
    
    def evaluate_and_visualize_thresholds(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                                         epoch: int, dataset_type: str = 'test',
                                         images_save_dir: str = None) -> Dict[str, float]:
        """
        Evaluate and create threshold visualization plots.
        
        Args:
            y_true: Ground truth batch
            y_pred: Predicted batch
            epoch: Training epoch
            dataset_type: 'train' or 'test'
            images_save_dir: Directory to save visualization images
            
        Returns:
            Dictionary of metrics
        """
        # Standard evaluation
        metrics = self.evaluate_and_log(y_true, y_pred, epoch, dataset_type, images_save_dir)
        
        # Create threshold visualizations if requested
        if images_save_dir and epoch % (self.save_images_frequency * 2) == 0:  # Less frequent than regular saves
            logger.info(f"Creating threshold visualizations for epoch {epoch}")
            
            # Threshold illustration
            thresh_save_path = f"{images_save_dir}/{dataset_type}_threshold_illustration_epoch_{epoch}.png"
            self.evaluator.plot_threshold_illustration(
                y_true, y_pred, 
                thresholds=[-0.5, 0.0, 0.5],
                save_path=thresh_save_path
            )
            
            # Fish detection analysis with practical metrics
            fish_analysis_save_path = f"{images_save_dir}/{dataset_type}_fish_analysis_epoch_{epoch}.png"
            recommendations = self.evaluator.plot_fish_detection_analysis(
                y_true, y_pred,
                save_path=fish_analysis_save_path
            )
            
            # 3-class confusion matrix
            cm_save_path = f"{images_save_dir}/{dataset_type}_3class_confusion_epoch_{epoch}.png"
            self.evaluator.plot_3class_confusion_matrix(
                y_true, y_pred,
                save_path=cm_save_path
            )
            
            logger.info(f"Fish analysis completed. Recommendations:")
            logger.info(f"  Current F1: {recommendations['current_f1']:.3f}, Max F1: {recommendations['max_f1']:.3f}")
            logger.info(f"  Optimal F1 threshold: {recommendations['optimal_f1_threshold']:.3f}")
            logger.info(f"  Optimal IoU threshold: {recommendations['optimal_iou_threshold']:.3f}")
            logger.info(f"  Precision-Recall AUC: {recommendations['precision_recall_auc']:.3f}")
        
        return metrics


def create_checkpoint_evaluator(checkpoint_dir: str, dataset_manager, generator_model):
    """
    Create a function to evaluate specific checkpoints.
    
    Usage:
        evaluate_checkpoint = create_checkpoint_evaluator(
            CHECKPOINT_DIR, dataset_manager, generator
        )
        
        # Evaluate latest checkpoint
        metrics = evaluate_checkpoint('latest')
        
        # Evaluate specific checkpoint
        metrics = evaluate_checkpoint('ckpt-1000')
    """
    
    def evaluate_checkpoint(checkpoint_name: str, num_samples: int = 20) -> Dict[str, float]:
        """
        Evaluate a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint ('latest' or specific like 'ckpt-1000')
            num_samples: Number of test samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load checkpoint
        checkpoint = tf.train.Checkpoint(generator=generator_model)
        
        if checkpoint_name == 'latest':
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"
        
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found at {checkpoint_dir}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint.restore(checkpoint_path)
        
        # Get test dataset
        _, test_ds, _ = dataset_manager.create_train_test_validation_datasets()
        
        evaluator = ImageEvaluator(threshold=0.0)
        all_metrics = []
        
        logger.info(f"Evaluating {num_samples} test samples...")
        
        for i, (input_batch, target_batch) in enumerate(test_ds.take(num_samples)):
            # Generate predictions
            pred_batch = generator_model(input_batch, training=False)
            
            # Evaluate this batch
            batch_metrics = evaluator.evaluate_batch(target_batch, pred_batch)
            all_metrics.append(batch_metrics)
            
            # Save some comparison images
            if i < 3:
                save_path = f"{checkpoint_dir}/eval_images/comparison_batch_{i}_{checkpoint_name}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                evaluator.plot_comparison(target_batch, pred_batch, 
                                        num_samples=min(3, target_batch.shape[0]),
                                        save_path=save_path)
        
        # Average metrics across all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([batch[key] for batch in all_metrics])
        
        # Print results
        print(f"\n=== CHECKPOINT EVALUATION: {checkpoint_name} ===")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name:25s}: {value:.4f}")
        print("=" * 60)
        
        return avg_metrics
    
    return evaluate_checkpoint


# =================== USAGE FUNCTIONS ===================

def evaluate_model_from_checkpoints(checkpoint_path: str, dataset_manager, 
                                   generator_model, num_samples: int = 50) -> Dict[str, float]:
    """
    Load a model checkpoint and evaluate it on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataset_manager: Your DatasetManager instance
        generator_model: Your generator model
        num_samples: Number of test samples to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator_model)
    checkpoint.restore(checkpoint_path)
    
    # Get test dataset
    _, test_ds, _ = dataset_manager.create_train_test_validation_datasets()
    
    evaluator = ImageEvaluator(threshold=0.0)  # Adjust threshold as needed
    
    all_metrics = []
    
    for i, (input_batch, target_batch) in enumerate(test_ds.take(num_samples)):
        # Generate predictions
        pred_batch = generator_model(input_batch, training=False)
        
        # Evaluate this batch
        batch_metrics = evaluator.evaluate_batch(target_batch, pred_batch)
        all_metrics.append(batch_metrics)
        
        # Visualize first few samples
        if i < 3:
            evaluator.plot_comparison(target_batch, pred_batch, 
                                    num_samples=min(3, target_batch.shape[0]),
                                    save_path=f'comparison_batch_{i}.png')
    
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([batch[key] for batch in all_metrics])
    
    return avg_metrics


def evaluate_test_images(test_dir: str, threshold: float = 0.0) -> Dict[str, float]:
    """
    Evaluate using saved test images (like in your test_images directory).
    
    Args:
        test_dir: Directory containing target_*.png and prediction_*.png files
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of evaluation metrics
    """
    import glob
    
    evaluator = ImageEvaluator(threshold=threshold)
    
    target_files = sorted(glob.glob(f"{test_dir}/target_*.png"))
    pred_files = sorted(glob.glob(f"{test_dir}/prediction_*.png"))
    
    if len(target_files) != len(pred_files):
        raise ValueError(f"Mismatch in number of target ({len(target_files)}) and prediction ({len(pred_files)}) files")
    
    all_targets = []
    all_preds = []
    
    for target_file, pred_file in zip(target_files, pred_files):
        # Load images
        target_img = plt.imread(target_file)
        pred_img = plt.imread(pred_file)
        
        # Convert to tensors and normalize to [-1, 1] if needed
        if target_img.max() <= 1.0:  # Already in [0, 1]
            target_tensor = tf.constant(target_img * 2.0 - 1.0, dtype=tf.float32)
            pred_tensor = tf.constant(pred_img * 2.0 - 1.0, dtype=tf.float32)
        else:  # In [0, 255]
            target_tensor = tf.constant(target_img / 127.5 - 1.0, dtype=tf.float32)
            pred_tensor = tf.constant(pred_img / 127.5 - 1.0, dtype=tf.float32)
        
        # Ensure proper shape [1, H, W, C]
        if len(target_tensor.shape) == 2:
            target_tensor = tf.expand_dims(tf.expand_dims(target_tensor, -1), 0)
            pred_tensor = tf.expand_dims(tf.expand_dims(pred_tensor, -1), 0)
        elif len(target_tensor.shape) == 3:
            target_tensor = tf.expand_dims(target_tensor, 0)
            pred_tensor = tf.expand_dims(pred_tensor, 0)
        
        all_targets.append(target_tensor)
        all_preds.append(pred_tensor)
    
    # Concatenate all images
    targets_batch = tf.concat(all_targets, axis=0)
    preds_batch = tf.concat(all_preds, axis=0)
    
    # Evaluate
    metrics = evaluator.evaluate_batch(targets_batch, preds_batch)
    
    # Visualize
    evaluator.plot_comparison(targets_batch, preds_batch, 
                            num_samples=min(5, targets_batch.shape[0]),
                            save_path=f'{test_dir}/evaluation_comparison.png')
    
    evaluator.plot_catch_overlay(targets_batch, preds_batch,
                               num_samples=min(3, targets_batch.shape[0]), 
                               save_path=f'{test_dir}/catch_overlay.png')
    
    evaluator.plot_confusion_matrix(targets_batch, preds_batch,
                                  save_path=f'{test_dir}/confusion_matrix.png')
    
    evaluator.plot_metrics_summary(metrics, 
                                 save_path=f'{test_dir}/metrics_summary.png')
    
    return metrics


# =================== EXAMPLE USAGE ===================

if __name__ == "__main__":
    import os
    
    # Example 1: Evaluate using test images directory
    print("=== Evaluating Test Images ===")
    if os.path.exists("/home/peder/fish-forecast/GAN-model/test_images"):
        test_metrics = evaluate_test_images("/home/peder/fish-forecast/GAN-model/test_images", threshold=0.0)
        
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric_name, value in test_metrics.items():
            print(f"{metric_name:25s}: {value:.4f}")
    else:
        print("Test images directory not found, skipping evaluation.")
    
    # Example 2: Quick evaluation with dummy data (for testing)
    print("\n=== Testing with Dummy Data ===")
    evaluator = ImageEvaluator(threshold=0.0)
    
    # Create dummy data in [-1, 1] range (like your GAN output)
    batch_size, height, width, channels = 4, 128, 128, 1
    y_true = tf.random.normal([batch_size, height, width, channels]) * 0.5  # Smaller range
    y_pred = y_true + tf.random.normal([batch_size, height, width, channels]) * 0.2  # Add noise
    
    # Evaluate
    dummy_metrics = evaluator.evaluate_batch(y_true, y_pred)
    
    print("\nDummy Data Results:")
    print("=" * 30)
    for metric_name, value in dummy_metrics.items():
        print(f"{metric_name:25s}: {value:.4f}")
    
    # Visualize dummy data
    evaluator.plot_comparison(y_true, y_pred, num_samples=2)