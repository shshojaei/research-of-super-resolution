#source: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

checkpoint_filepath = 'drive/MyDrive/Colab Notebooks/EDSR/checkpoint'
STEPS_PER_EPOCH = 200

#Create a callback that saves the model's weights every 10 epochs 
#source: https://stackoverflow.com/questions/59069058/save-model-every-10-epochs-tensorflow-keras-v2

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    verbose=1,
    monitor='val_loss',
    mode='min',
    save_freq= int(10 * STEPS_PER_EPOCH)
)


model.compile(optimizer=optimizer , loss='mse' , metrics=[PSNR])

history = model.fit(train_ds, epochs=50, steps_per_epoch=200, validation_data=val_ds, callbacks=[model_checkpoint_callback])
