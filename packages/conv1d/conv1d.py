#!/usr/bin/env python3

import torch.nn.functional as F
import torch
import numpy as np
import yaml
import sys
import os


def main():
	inputs = torch.randn(33, 16, 30)
	filters = torch.randn(20, 16, 5)
	result = F.conv1d(inputs, filters)
	
	return result

if __name__ == '__main__':
	main()
