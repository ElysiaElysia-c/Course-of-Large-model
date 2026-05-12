# 自注意力机制（Self-Attention）基础实现
import torch
I = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
)
#I 是一个6×3 的矩阵
#6 行：代表一句话里的 6 个单词（Your journey starts with one step）
#3 列：代表每个单词的词向量维度（3 维向量表示一个词）
A = I @ I.T   # 或A = torch.matmul(I, I.T)         计算词向量之间的相似度 点积越大，两个词越相似。
print("初始注意力权重矩阵：") 
print(A)
# SoftMax归一化  softmax：归一化函数，把任意数值转换成 0~1 之间的概率，且每一行的和为 1
A_dot = torch.softmax(A, dim=1)  # dim=0则按列， 用 Softmax 把权重归一化
print("归一化后注意力权重矩阵：") 
print(A_dot)

# A_dot与I相乘
O = A_dot @ I    # 或O = torch.matmul(A_dot, I)
print("输入向量矩阵I：")
print(I)
print("输出向量矩阵O：")
print(O)
# 其他归一化
row_sums = A.sum(dim=1, keepdim=True)
A_dot = A / row_sums
print("归一化后注意力权重矩阵：")
print(A_dot)

#实现自注意力机制
d = I.shape[1] 
dk = 2 
print("Input vector dimension:", d)
print("Q\K\V vector dimension:", dk)
torch.manual_seed(123)
Q = torch.nn.Parameter(torch.rand(d, dk), requires_grad=True)
K   = torch.nn.Parameter(torch.rand(d, dk), requires_grad=True)
V = torch.nn.Parameter(torch.rand(d, dk), requires_grad=True)
#requires_grad=True的时候，这个张量参与前向传播后，反向传播时会计算它的梯度，梯度会存储在张量的.grad属性里，
# 而且优化器在更新参数时会用到这些梯度。
# 反之，如果requires_grad=False，那么这个张量不会被跟踪梯度，反向传播时不会计算它的梯度，
# 这样可以节省内存和计算资源，通常用于不需要更新的参数（比如固定的权重、或者输入数据）
# 将Q与转置K相乘,得到初始注意力矩阵A
A = Q @ K.T    # 或A = torch.matmul(Q, K.T)
print("初始注意力权重矩阵：")
print(A)
# SoftMax归一化,按行
A_dot = torch.softmax(A/dk**0.5, dim=1)  # dim=0则按列
print("归一化后注意力权重矩阵：")
print(A_dot)
# print("归一化按行求和：")
# print(A_dot.sum(dim=1))

# A_dot与V相乘
O = A_dot @ V    # 或O = torch.matmul(A_dot, V)
print("输入向量矩阵I：")
print(I)
print("输出向量矩阵O：")
print(O)



