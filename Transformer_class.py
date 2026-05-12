#封装为自注意力的类
import torch
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d, dk):
        super().__init__()
        self.dk = dk
        self.W_query = nn.Parameter(torch.rand(d, dk))
        self.W_key   = nn.Parameter(torch.rand(d, dk))
        self.W_value = nn.Parameter(torch.rand(d, dk))

    def forward(self, I):
        K = I @ self.W_key
        Q = I @ self.W_query
        V = I @ self.W_value
        A = Q @ K.T
        A_dot = torch.softmax(A/ K.shape[-1]**0.5, dim=-1)
        O = A_dot @ V
        return O
    
I = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
)

d = I.shape[1] 
dk = 2 
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d, dk)
print(sa_v1(I))
