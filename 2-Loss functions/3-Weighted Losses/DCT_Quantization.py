# Quantization matrix
Q = tf.constant([[16, 11, 10, 16, 24, 40, 51, 61],
                 [12, 12, 14, 19, 26, 58, 60, 55],
                 [14, 13, 16, 24, 40, 57, 69, 56],
                 [14, 17, 22, 29, 51, 87, 80, 62],
                 [18, 22, 37, 56, 68, 109, 103, 77],
                 [24, 35, 55, 64, 81, 104, 113, 92],
                 [49, 64, 78, 87, 103, 121, 120, 101],
                 [72, 92, 95, 98, 112, 100, 103, 99]])
Q = tf.cast(Q, dtype=tf.float32)
# Reshape Q to match y_true_dct dimensions
Q_4d = tf.tile(Q, [24,24])
#print(Q_4d)
Q_4d = tf.tile(tf.reshape(Q_4d, [1, 192, 192, 1]), [1, 1, 1, 3])
#print(Q_4d)

def dct_Q_loss(y_true, y_pred):

  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)

  y_true_dct = tf.signal.dct(y_true)
  y_pred_dct = tf.signal.dct(y_pred)

  y_true_quant = y_true_dct / Q_4d
  y_pred_quant = y_pred_dct / Q_4d

  return tf.reduce_mean(tf.abs(y_true_quant - y_pred_quant))

#############################################################

import tensorflow as tf
import numpy as np

def DCT_loss(y_true, y_pred):

    y_true = tf.Variable(y_true)
    y_pred = tf.Variable(y_pred)

    # Convert the input tensors to unsigned 8-bit integers
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute the DCT of the true and predicted tensors for each color channel
    true_dct = tf.stack([tf.signal.dct(tf.cast(y_true[..., i], tf.float32), type=2, norm='ortho') for i in range(3)], axis=-1)
    pred_dct = tf.stack([tf.signal.dct(tf.cast(y_pred[..., i], tf.float32), type=2, norm='ortho') for i in range(3)], axis=-1)

    # Quantize the DCT coefficients for each color channel using the STE technique
    Q = tf.constant([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68,109,103, 77],
                   [24, 35, 55, 64, 81,104,113, 92],
                   [49, 64, 78, 87,103,121,120,101],
                   [72, 92, 95, 98,112,100,103, 99]])
    Q = tf.cast(Q, dtype=tf.float32)

    quantize_dct_true = tf.stack([tf.keras.backend.round(true_dct[..., i] / Q) * Q for i in range(3)], axis=-1)
    quantize_dct_pred = tf.stack([tf.keras.backend.round(pred_dct[..., i] / Q) * Q for i in range(3)], axis=-1)

    # Compute the L1 distance between the DCT coefficients for each color channel
    loss = tf.reduce_mean(tf.abs(quantize_dct_true - quantize_dct_pred), axis=[1, 2])

    return loss
