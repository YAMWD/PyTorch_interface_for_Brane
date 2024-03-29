import torch
filters = torch.randn(33, 16, 3, 3, 3)
torch.save(filters, 'weights_conv3d.pt')