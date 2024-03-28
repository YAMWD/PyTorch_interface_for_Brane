import torch
inputs = torch.randn(20, 16, 50, 10, 20)
torch.save(inputs, 'inputs_conv3d.pt')