import tensorflow as tf
import keras.backend as K

def dssim_main(img1, img2, max_val=255):

    img1 = K.cast( img1, 'float32')
    img2 = K.cast( img2, 'float32')

    # Calculate mean of img1 and img2 across all channels
    mu1 = tf.reduce_mean(img1, axis=[1, 2, 3], keepdims=True)
    mu2 = tf.reduce_mean(img2, axis=[1, 2, 3], keepdims=True)

    # Calculate variance of img1 and img2 across all channels
    var1 = tf.reduce_mean(tf.square(img1 - mu1), axis=[1, 2, 3], keepdims=True)
    var2 = tf.reduce_mean(tf.square(img2 - mu2), axis=[1, 2, 3], keepdims=True)

    # Calculate covariance between img1 and img2 across all channels
    covar = tf.reduce_mean((img1 - mu1) * (img2 - mu2), axis=[1, 2, 3], keepdims=True)

    # SSIM constants
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    c3 = c2/2

    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * covar + c2)
    denominator = (tf.square(mu1) + tf.square(mu2) + c1) * (var1 + var2 + c2)
    ssim = numerator / denominator

    # Average SSIM across all channels
    ssim_mean = tf.reduce_mean(ssim, axis=[1, 2, 3])

    return 1-ssim_mean
