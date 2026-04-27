import tensorflow as tf
from skimage.metrics import hausdorff_distance as skimage_hausdorff

# TODO: Add PyTorch equivalents for dice_coef, soft_dice_loss, bce_dice_loss, and hausdorff_distance.

# loss function
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