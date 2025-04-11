import torch

torch.cuda.is_available()
# -> True

torch.tensor([0.1, 0.2], device=torch.device('cuda:0'))
# -> tensor([0.1000, 0.2000], device='cuda:0')