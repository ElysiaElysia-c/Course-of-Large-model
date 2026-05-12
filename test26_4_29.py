import torch
import torch.nn as nn

I = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
    [0.55, 0.87, 0.66],  # journey  (x^2)
    [0.57, 0.85, 0.64],  # starts   (x^3)
    [0.22, 0.58, 0.33],  # with     (x^4)
    [0.77, 0.25, 0.10],  # one      (x^5)
    [0.05, 0.80, 0.55]]  # step     (x^6)
)


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, I):
        K = self.W_key(I)
        Q = self.W_query(I)
        V = self.W_value(I)
        A = Q @ K.T
        A_dot = torch.softmax(A / self.d_out ** 0.5, dim=-1)
        O = A_dot @ V
        return O


d = I.shape[1]
dk = 2

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d, dk)
print(sa_v2(I))

Q = sa_v2.W_query(I)
K = sa_v2.W_key(I)
A = Q @ K.T
A_dot = torch.softmax(A / K.shape[-1] ** 0.5, dim=1)
print(A_dot)

d = A_dot.shape[0]
mask = torch.tril(torch.ones(d, d))
print(mask)
masked_Adot = A_dot * mask
print(masked_Adot)

row_sums = masked_Adot.sum(dim=1, keepdim=True)
print(row_sums)
masked_attentions = masked_Adot / row_sums
print(masked_attentions)

masked_attentions = torch.softmax(masked_Adot, dim=1)
print(masked_Adot)
print(masked_attentions)

Q = sa_v2.W_query(I)
K = sa_v2.W_key(I)
A = Q @ K.T
A_dot = torch.softmax(A / K.shape[-1] ** 0.5, dim=1)
print(A_dot)

d = A_dot.shape[0]
mask = torch.triu(torch.ones(d, d), diagonal=1)  # 对角线以及对角线以上 置为0
print(mask)
masked_Adot = A_dot.masked_fill(mask.bool(), -torch.inf)
print(masked_Adot)

masked_attentions = torch.softmax(masked_Adot, dim=1)
print(masked_attentions)

#信息泄露
Q = sa_v2.W_query(I)
K = sa_v2.W_key(I)
A = Q @ K.T
# A_dot = torch.softmax(A / K.shape[-1]**0.5, dim=1)
# print(A_dot)
d = A.shape[0]
mask = torch.triu(torch.ones(d,d), diagonal=1)
# masked_Adot = A_dot.masked_fill(mask.bool(), -torch.inf)
masked_A = A.masked_fill(mask.bool(), -torch.inf)
print(masked_A)
masked_attentions = torch.softmax(masked_A / K.shape[-1]**0.5, dim=1)
print(masked_attentions)

# drop out 的使用
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # 使用的dropout率为0.5
example = torch.ones(6, 6)  # 创建一个由1组成的矩阵
print(dropout(example))
print(dropout(example))
print(dropout(example))

masked_attentions = torch.softmax(masked_A / K.shape[-1]**0.5, dim=1)
print(masked_attentions)
dropout = torch.nn.Dropout(0.5)  # 使用的dropout率为0.5
print(dropout(masked_attentions))

# 避免信息泄露+使用DROPOUT
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, X):
        b, num_tokens, d_in = X.shape
        K = self.W_key(X)
        Q = self.W_query(X)
        V = self.W_value(X)
        A = Q @ K.transpose(1, 2)
        A.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = self.dropout(torch.softmax(A/self.d_out**0.5, dim=-1))
        O = attn_weights @ V
        return O


X = torch.stack((I, I), dim=0)
print(X.shape)
print(X)

b, context_length, d_in = X.shape
d_out = 2
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(X)
print("context_vecs.shape:", context_vecs.shape)
print(context_vecs)

# 多头注意力
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(X)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                #使用线性层组合头部输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)  # C
        queries = self.W_query(x)  # C
        values = self.W_value(x)  # C

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # D

        keys = keys.transpose(1, 2)  # E
        queries = queries.transpose(1, 2)  # E
        values = values.transpose(1, 2)  # E

        attn_scores = queries @ keys.transpose(2, 3)  # F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # G

        attn_scores.masked_fill_(mask_bool, -torch.inf)  # H

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  # I

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # J
        context_vec = self.out_proj(context_vec)  # K
        return context_vec

# C 张量形状：(b, num_tokens, d_out)
# D 我们通过添加 num_heads 维度来隐式地拆分矩阵。然后展开最后一个维度，使其形状从 (b, num_tokens, d_out) 转换为 (b, num_tokens, num_heads, head_dim)
# E 将张量的形状从 (b, num_tokens, num_heads, head_dim) 转置为 (b, num_heads, num_tokens, head_dim)
# F 对每个注意力头进行点积运算
# G 掩码被截断到 token 的数量
# H 使用掩码填充注意力分数
# I 张量形状：（b, num_tokens, n_heads, head_dim）
# J 将多个注意力头的输出结果合并，其中输出维度 self.d_out 等于注意力头数 self.num_heads 与每个头的维度 self.head_dim 的乘积
# K 添加一个可选的线性投影层

torch.manual_seed(123)
batch = torch.stack((I, I), dim=0)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)




