# Custom ResNet-Inspired CNN for CIFAR-10

This project implements a custom **ResNet-inspired Convolutional Neural Network** from scratch using PyTorch for image classification on the CIFAR-10 dataset.

Unlike the official ResNet-18 architecture, this implementation explores an alternative residual block design by using **MaxPooling for downsampling in the main branch** while using a **1×1 projection shortcut** to match feature dimensions.

The goal of this project was to understand residual learning and CNN architectures beyond simply using pretrained models.

---

## Final Results

| Metric                | Value                            |
| --------------------- | -------------------------------- |
| Dataset               | CIFAR-10                         |
| Epochs                | 50                               |
| Optimizer             | Adam                             |
| Initial Learning Rate | 0.001                            |
| Scheduler             | StepLR (step_size=15, gamma=0.1) |
| Weight Decay          | 0.0001                           |
| Batch Size            | 32                               |
| Train Accuracy        | **96.86%**                       |
| Test Accuracy         | **92.41%**                       |

---

## Architecture Overview

```text
Input Image (32×32×3)
        │
        ▼
Conv(3 → 64)
        │
        ▼
Residual Stage 1
├── ResBlock(64,64)
└── ResBlock(64,64)

        ▼

Residual Stage 2
├── ResBlock(64,128)
└── ResBlock(128,128)

        ▼

Residual Stage 3
├── ResBlock(128,256)
└── ResBlock(256,256)

        ▼

Residual Stage 4
├── ResBlock(256,512)
└── ResBlock(512,512)

        ▼

Adaptive Average Pooling

        ▼

Fully Connected (512)

        ▼

Dropout (0.2)

        ▼

Output Layer (10 Classes)
```

---

## Custom Residual Block

Each residual block consists of:

* 3×3 Convolution
* Batch Normalization
* ReLU Activation
* Optional MaxPooling (for downsampling)
* Second 3×3 Convolution
* Projection Shortcut using 1×1 Convolution
* Residual Addition

Downsampling blocks use:

```text
Main Branch:
Conv3×3
   ↓
BatchNorm
   ↓
ReLU
   ↓
MaxPool2d
   ↓
Conv3×3

Shortcut Branch:
1×1 Conv (stride=2)
   ↓
BatchNorm
```

The projection shortcut ensures that tensor dimensions match before residual addition.

---

## Data Augmentation

Training images were augmented using:

* Random Crop (padding=4)
* Random Horizontal Flip
* Random Rotation (10°)

Test images were evaluated without augmentation.

---

## Training Configuration

```python
Optimizer      : Adam
Learning Rate  : 0.001
Weight Decay   : 0.0001
Scheduler      : StepLR(step_size=15, gamma=0.1)
Loss Function  : CrossEntropyLoss
Epochs         : 50
Batch Size     : 32
```

---

## Concepts Implemented

* Convolutional Neural Networks
* Residual Learning
* Skip Connections
* Projection Shortcuts
* Batch Normalization
* Adaptive Average Pooling
* Data Augmentation
* Learning Rate Scheduling
* Weight Decay Regularization

---

## Motivation

Rather than using a pretrained model, this project was built to better understand how residual networks work internally.

The architecture, residual blocks, and training pipeline were implemented manually to gain a deeper understanding of:

* Forward propagation
* Residual connections
* Gradient flow
* CNN training dynamics
* Hyperparameter tuning

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## Future Improvements

* Experiment with SGD + Momentum.
* Compare against the official ResNet-18 implementation.
* Add CutMix and MixUp augmentation.
* Visualize learned feature maps.
* Train on larger datasets such as Tiny ImageNet.

---

## Author

**Vivek**

B.Tech Computer Science and Engineering (AI & ML)

VIT-AP University
