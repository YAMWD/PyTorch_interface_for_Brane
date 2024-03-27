#!/usr/bin/env python3

import torch.nn.functional as F
import torch
import numpy as np
import yaml
import sys
import os

def readTensor(filepath):

	# result = torch.load(filepath)

	result = filepath

	return result

def conv1d(inputs, weights):
	result = F.conv1d(inputs, weights)

	return result

def main():
    # Make sure that at least one argument is given
	'''
    if len(sys.argv) != 2 or (sys.argv[1] != "readTensor" and sys.argv[1] != "conv1d"):
        print(f"Usage: {sys.argv[0]} readTensor | conv1d")
        exit(1)
	'''

    # If it checks out, call the appropriate function
    command = sys.argv[1]
    if command == "readTensor":
        # Parse the input as JSON, then pass that to the `read_tensor` function
        inputs = json.loads(os.environ["INPUTS"])
        result = readTensor(inputs)

	'''
    else:
        # Parse the input as JSON, then pass that to the `conv1d` function
        inputs = json.loads(os.environ["INPUTS"])
        weights = json.loads(os.environ["WEIGHTS"])
        result = conv1d(inputs, weights)
	'''
    # Print the result with the YAML package
    print(yaml.dump({ "output": result }))

if __name__ == '__main__':
	main()
