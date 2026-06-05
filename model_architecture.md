# Model Architecture

This project implements a custom ResNet-inspired CNN using manually designed residual blocks.

## Residual Block

Each block consists of:

* 3×3 Convolution
* Batch Normalization
* ReLU Activation
* Optional MaxPool2d downsampling
* Second 3×3 Convolution
* Projection shortcut using a 1×1 convolution
* Residual addition

Projection shortcuts are used whenever the input and output dimensions differ.

## Network Structure

* Initial Conv Layer (3 → 64)
* 2 Residual Blocks (64)
* 2 Residual Blocks (128)
* 2 Residual Blocks (256)
* 2 Residual Blocks (512)
* Adaptive Average Pooling
* Fully Connected Layer (512)
* Dropout (0.2)
* Output Layer (10 classes)
