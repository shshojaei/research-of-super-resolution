I'm using `PyTorch 1.4` in `Python 3.6`.


code from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution


The main code trained the model with the "imagenet" dataset and 1000000 epochs, but here the model is training with the "DIV2K" dataset and 53 epochs.


We're going to be implementing [_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_](https://arxiv.org/abs/1609.04802). It's not just that the results are very impressive... it's also a great introduction to GANs!


We will train the two models described in the paper — the SRResNet, and the SRGAN which greatly improves upon the former through adversarial training.  

