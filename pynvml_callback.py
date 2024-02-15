#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:33:16 2020

@author: ing
"""

import tensorflow as tf
from pynvml import *

class GPUMonitorCallback(tf.keras.callbacks.Callback):
    """
    A TensorFlow callback to monitor GPU and memory usage after a specified number of epochs.

    Attributes:
        frequency (int): Specifies after how many epochs the GPU stats should be printed.
    """
    def __init__(self, frequency=10):
        """
        Initializes the GPUMonitorCallback instance.

        Parameters:
            frequency (int): Frequency in epochs to check and print GPU stats.
        """
        super(GPUMonitorCallback, self).__init__()
        self.frequency = frequency  # Set the frequency for monitoring GPU stats.

    def on_epoch_end(self, epoch, logs=None):
        """
        Override the on_epoch_end method to print GPU stats at the end of specified epochs.

        Parameters:
            epoch (int): The index of the epoch that just ended.
            logs (dict): A dictionary of logs results from the training epoch.
        """
        if epoch % self.frequency == 0:  # Check if it's time to print GPU stats based on the frequency.
            self.print_gpu_stats()

    def print_gpu_stats(self):
        """
        Fetches and prints the GPU utilization and memory usage for all available GPUs.
        """
        nvmlInit()  # Initialize NVML to interact with the NVIDIA driver.
        deviceCount = nvmlDeviceGetCount()  # Get the count of available NVIDIA GPUs.
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)  # Get the handle for the ith GPU.
            info = nvmlDeviceGetMemoryInfo(handle)  # Fetch memory information.
            util = nvmlDeviceGetUtilizationRates(handle)  # Fetch utilization rates.
            
            # Print the GPU device name, memory usage, and GPU utilization.
            print(f"Device {i}: {nvmlDeviceGetName(handle)}")
            print(f"Memory Usage: {info.used // (1024 ** 2)} / {info.total // (1024 ** 2)} MB")
            print(f"GPU Utilization: {util.gpu}%")
        nvmlShutdown()  # Shutdown NVML to clean up resources.

        