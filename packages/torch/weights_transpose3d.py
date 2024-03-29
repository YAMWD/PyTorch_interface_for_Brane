import torch
weights = torch.randn(16, 33, 3, 3, 3)
torch.save(weights, 'weights_transpose3d.pt')