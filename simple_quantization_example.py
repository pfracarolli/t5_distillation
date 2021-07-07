import torch

W = torch.tensor([
    [-0.18120981, -0.29043840],
    [0.49722983, 0.22141714]
])

x = torch.tensor([
    [0.77412377],
    [0.49299395]
])

y = torch.matmul(W, x)
# tensor([[-0.2835],
#         [ 0.4941]])

W_q = torch.floor(128 * W).to(torch.int16)
x = torch.floor(128 * x).to(torch.int16)

y_q = torch.matmul(W_q, x)
# tensor([[-4770],
#         [ 8001]])

y = y_q.to(torch.float32) / 16384  
# tensor([[-0.2911],
#         [ 0.4883]])