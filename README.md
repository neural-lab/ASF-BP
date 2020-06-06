# ASF-BP

The repo contains the code associated with the SNN training method ASF-BP. The code has been tested with Pytorch 1.1.0 abd Python 3.7.4.

## Testing and Training 
To train a new model from scratch, the basic syntax is: ```python cifar10_vgg7-ASF.py```

To test a pre-trained model, the basic syntax is ```python cifar10_ResNet11.py --resume model_bestT1_cifar10_r11.pth.tar --evaluate```

## Reference
Chankyu Lee, Syed Shakib Sarwar, Priyadarshini Panda, Gopalakrishnan Srinivasan, and Kaushik Roy. Enabling spike-based backpropagation for training deep neural network architectures. Frontiers in Neuroscience, 14, 2020.

