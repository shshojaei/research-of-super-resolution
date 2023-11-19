!pip install livelossplot

from livelossplot import PlotLossesKeras

history = model.fit(train_ds, epochs=100, steps_per_epoch=200, validation_data=val_ds , callbacks=[PlotLossesKeras()])
