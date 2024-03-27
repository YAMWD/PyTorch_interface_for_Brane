#!/usr/bin/env python3

import torch.nn.functional as F
import torch
import numpy as np
import yaml
import sys
import os
import json

def conv1d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, dilation = 1, groups = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv1d(inputs, weights, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def conv2d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, dilation = 1, groups = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv2d(inputs, weights, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def main():
    # Make sure that at least one argument is given
    if len(sys.argv) != 2 or (sys.argv[1] != "conv1d" and sys.argv[1] != "conv2d"):
        print(f"Usage: {sys.argv[0]} conv1d | conv2d")
        exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "conv1d":
        # Parse the input as JSON, then pass that to the `conv1d` function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        dilation = json.loads(os.environ["DILATION"])
        groups = json.loads(os.environ["GROUPS"])
        result = conv1d(inputs_path, weights_path, bias_path, stride, padding, dilation, groups)
    else:
        # Parse the input as JSON, then pass that to the `conv2d` function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        dilation = json.loads(os.environ["DILATION"])
        groups = json.loads(os.environ["GROUPS"])
        result = conv2d(inputs_path, weights_path, bias_path, stride, padding, dilation, groups)

if __name__ == '__main__':
	main()
