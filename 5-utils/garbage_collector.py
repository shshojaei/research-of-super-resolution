#source: https://stackoverflow.com/questions/61188185/how-to-free-memory-in-colab

# Garbage Collector - use it like gc.collect()
import gc

# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

saver = GarbageCollectorCallback()
