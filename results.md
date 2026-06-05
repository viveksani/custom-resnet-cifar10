# Training Results

## Configuration

* Dataset: CIFAR-10
* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Scheduler: StepLR(step_size=15, gamma=0.1)
* Weight Decay: 0.0001
* Batch Size: 32

## Performance

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 96.86% |
| Test Accuracy  | 92.41% |

## Data Augmentation

* Random Crop
* Random Horizontal Flip
* Random Rotation
