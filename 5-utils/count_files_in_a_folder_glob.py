from glob import glob
files = sorted(glob('./DIV2K_train_HR/*.npy'))
print(len(files))
