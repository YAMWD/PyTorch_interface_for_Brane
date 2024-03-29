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

def conv3d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, dilation = 1, groups = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv3d(inputs, weights, bias = bias, stride = stride, padding = padding, dilation = dilation, groups = groups)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def conv_transpose1d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv_transpose1d(inputs, weights, bias = bias, stride = stride, padding = padding, output_padding = output_padding, groups = groups, dilation = dilation)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def conv_transpose2d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv_transpose2d(inputs, weights, bias = bias, stride = stride, padding = padding, output_padding = output_padding, groups = groups, dilation = dilation)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def conv_transpose3d(inputs_path, weights_path, bias = "None", stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1):
      
    inputs = torch.load(inputs_path)
    weights = torch.load(weights_path)

    if bias == "None":
        bias = None
    else:
        bias = torch.load(bias)

    result = F.conv_transpose2d(inputs, weights, bias = bias, stride = stride, padding = padding, output_padding = output_padding, groups = groups, dilation = dilation)

    filepath = 'result.pt'
    torch.save(result, filepath)

    return filepath

def main():
    # Make sure that at least one argument is given
    if len(sys.argv) != 2:
        print("Usage: python code.py <command>")
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
    elif command == "conv2d":
        # Parse the input as JSON, then pass that to the `conv2d` function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        dilation = json.loads(os.environ["DILATION"])
        groups = json.loads(os.environ["GROUPS"])
        result = conv2d(inputs_path, weights_path, bias_path, stride, padding, dilation, groups)
    elif command == "conv3d":
        # Parse the input as JSON, then pass that to the `conv3d` function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        dilation = json.loads(os.environ["DILATION"])
        groups = json.loads(os.environ["GROUPS"])
        result = conv2d(inputs_path, weights_path, bias_path, stride, padding, dilation, groups)
    elif command == "conv_transpose1d":
        # Parse the input as JSON, then pass that to the 'conv_transpose1d' function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        output_padding = json.loads(os.environ["OUTPUT_PADDING"])
        groups = json.loads(os.environ["GROUPS"])
        dilation = json.loads(os.environ["DILATION"])
        result = conv_transpose1d(inputs_path, weights_path, bias_path, stride, padding, output_padding, groups, dilation)
    elif command == "conv_transpose2d":
        # Parse the input as JSON, then pass that to the 'conv_transpose2d' function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        output_padding = json.loads(os.environ["OUTPUT_PADDING"])
        groups = json.loads(os.environ["GROUPS"])
        dilation = json.loads(os.environ["DILATION"])
        result = conv_transpose2d(inputs_path, weights_path, bias_path, stride, padding, output_padding, groups, dilation)
    elif command == "conv_transpose3d":
        # Parse the input as JSON, then pass that to the 'conv_transpose3d' function
        inputs_path = json.loads(os.environ["INPUTS"])
        weights_path = json.loads(os.environ["WEIGHTS"])
        bias_path = json.loads(os.environ["BIAS"])
        stride = json.loads(os.environ["STRIDE"])
        padding = json.loads(os.environ["PADDING"])
        output_padding = json.loads(os.environ["OUTPUT_PADDING"])
        groups = json.loads(os.environ["GROUPS"])
        dilation = json.loads(os.environ["DILATION"])
        result = conv_transpose2d(inputs_path, weights_path, bias_path, stride, padding, output_padding, groups, dilation)


if __name__ == '__main__':
	main()
