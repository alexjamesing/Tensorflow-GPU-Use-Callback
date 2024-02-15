# GPU Monitor Callback for TensorFlow/Keras

## Overview
This TensorFlow/Keras callback (`GPUMonitorCallback`) is designed to monitor and notify you of the GPU usage during the training process. It provides insights into the GPU utilization and memory usage after a specified number of epochs, helping you understand how effectively your model training is utilizing the GPU.

## Features
- **GPU Utilization**: Monitors the percentage of GPU utilization.
- **Memory Usage**: Monitors the GPU memory usage in megabytes.
- **Frequency Control**: Allows you to set how frequently (in terms of epochs) the GPU stats should be printed.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- You have installed TensorFlow and it is properly configured to use your GPU.
- You have installed the Python NVIDIA management library (`pynvml`).

## Usage
To use the `GPUMonitorCallback`, follow these steps:

1. Import the callback into your training script:

```python
from GPUMonitorCallback import GPUMonitorCallback
```
Instantiate the callback, optionally setting the frequency of epochs:

```
gpu_monitor = GPUMonitorCallback(frequency=5)
```

Add the callback to your model's fit method:

```
model.fit(x_train, y_train, epochs=50, callbacks=[gpu_monitor])
```

Now, when you train your model, you will see the GPU usage statistics printed out every 5 epochs.

## Contributing

Contributions to this callback are welcome. If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT License. See LICENSE for more information.



