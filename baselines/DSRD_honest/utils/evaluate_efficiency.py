#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for tracking training efficiency."""
import torch
import time
import numpy as np

class evaluate_efficiency:
    
    def __init__(self, device="cuda:0"):
        
        
        self.device = device
        self.latency = []           # ms per epoch
        self.gpu_peak_memory = []   # MB
        self.throughput = []        # samples/ms
        self.epoch = 0
        self.reset()
        
        
    def reset(self):
        self.num_samples =  0
        self.runing_time = 0
        self.gpu_memory = []
        self.epoch += 1
        
    def start(self, samples):
        
        self.num_samples += len(samples)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.time()*1000
    
    
    def end(self):
        
        self.end_time = time.time()*1000
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        self.gpu_memory.append(peak_memory)
        
        time_interval = self.end_time - self.start_time
        self.runing_time += time_interval
    
    def caculate(self):
        '''caculate metrics at the end of each epoch'''
        
        latency = self.runing_time
        gpu_peak_memory = np.mean(self.gpu_memory)
        throughput = self.num_samples/self.runing_time
        
        self.latency.append(latency)
        self.gpu_peak_memory.append(gpu_peak_memory)
        self.throughput.append(throughput)
        self.reset()

        
    def return_metrics(self):
        average_latency = np.mean(self.latency)
        average_gpu_memory = np.mean(self.gpu_peak_memory)
        average_throughput = np.mean(self.throughput)
        return average_latency,average_gpu_memory,average_throughput
