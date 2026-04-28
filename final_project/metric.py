import tensorflow as tf
from skimage.metrics import hausdorff_distance as skimage_hausdorff
import numpy as np

# Monte carlo dropout for uncertanty estimation
def mc_prediction(model, x, num_passes):
    """
    Run MC dropout inference on input data. Performs multiple forward passes
    with dropout enabled, allowing us to estimate prediction uncertainty.

    Parameters:
        model: Keras model with Dropout layers
        x: Input data
        num_passes: number of forward passes to perform (default: 50)
            The more passes, the better uncertainty estimate but it will be slower
    Returns:
        mean_prediction: the mean of all the predictions
        std_prediction: the standard deviation of the predictions
        all_predictions: the list of all predictions
    """
    predictions = []

    for i in range(num_passes):
        # Forward passes
        prediction = model(x, training=False)
        predictions.append(prediction.numpy())
        
        print(f"Monte Carlo pass {i + 1}/{num_passes} completed")
    
    # Convert to numpy and calculate mean and std
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis = 0)
    std_prediction = np.std(predictions, axis = 0)

    return mean_prediction, std_prediction, predictions

def uncertainty_map(std_prediction, threshold = 0.15):
    """
    Create an uncertainty map showing which predictions are unreliable given a threshold

    Parameters:
        std_prediction: Standard deviation from Monte Carlo Dropout
        threshold: Pixels with standard deviation above this value are considered uncertain
    
    Returns:
        uncertainty_binary: Binary map where 1 = uncertain, 0 = confident
    """
    uncertainty_binary = (std_prediction > threshold).astype(np.float32)
    return uncertainty_binary

# Loss functions and metrics
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denom + smooth)


@tf.function
def soft_dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))

# sum of bce and dice loss
@tf.function
def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + soft_dice_loss(y_true, y_pred)

# hausdorff distance
@tf.function
def hausdorff_distance(y_true, y_pred):
    """
    Compute the Hausdorff distance between two binary masks using scikit-image.
    """
    # Inner helper function to run outside Tensorflow graph
    def _hausdorff(y_true, y_pred):
        # Converts y_true and y_pred to binary masks (anything > 0.5 becomes True, False otherwise) and converts it to a NumPy array
        y_true = (y_true > 0.5).numpy()
        y_pred = (y_pred > 0.5).numpy()

        # Call scikit-image on Hausdorff distance on the binary numpy arrays
        return skimage_hausdorff(y_true, y_pred)
    
    # Wraps the _hausdorff to run withing TF's graph (casts inputs to float32, executes the function, and returns it as float64)
    result = tf.py_function(_hausdorff, [tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)], tf.float64)
    return tf.cast(result, tf.float32) # Converts back to float32 for consistency with other metrics