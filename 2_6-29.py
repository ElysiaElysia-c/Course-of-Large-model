from importlib.metadata import version
import tiktoken
# print("tiktoken version:", version("tiktoken")) # 输出 tiktoken 的版本信息，确保使用的是正确的版本。
tokenizer = tiktoken.get_encoding("gpt2") # 获取 GPT-2 模型使用的编码器，这个编码器会将文本转换为 token ID

with open("The_Verdict.txt", "r", encoding="utf-8") as f: 
      raw_text = f.read() 
enc_text = tokenizer.encode(raw_text) # 使用 tiktoken 的 encode方法将原始文本转换为 token ID 列表
print(len(enc_text))
enc_sample = enc_text[50:] # 移除前50个token
print(len(enc_sample))

context_size = 4                    #A
x = enc_sample[:context_size]  #x包含输入token，y包含目标
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:    {y}")

for i in range(1, context_size+1):
      context = enc_sample[:i]
      desired = enc_sample[i]
      print(context, "---->", desired)

for i in range(1, context_size+1):
      context = enc_sample[:i]
      desired = enc_sample[i]
      print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

import torch
from torch.utils.data import Dataset, DataLoader
# 数据加载器类
class GPTDatasetV1(Dataset):
      def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            token_ids = tokenizer.encode(txt)    #A 将整个文本进行分词

            for i in range(0, len(token_ids) - max_length, stride):   #B 使用滑动窗口将书籍分块为最大长度的重叠序列。
                  input_chunk = token_ids[i:i + max_length]
                  target_chunk = token_ids[i + 1: i + max_length + 1]
                  self.input_ids.append(torch.tensor(input_chunk))
                  self.target_ids.append(torch.tensor(target_chunk))

      def __len__(self):       #C 返回数据集的总行数
            return len(self.input_ids)

      def __getitem__(self, idx):    #D 从数据集中返回指定行
            return self.input_ids[idx], self.target_ids[idx]


# 加载数据
def create_dataloader_v1(txt, batch_size=4, max_length=256,
      stride=128, shuffle=True, drop_last=True, num_workers=0):
      tokenizer = tiktoken.get_encoding("gpt2")    #A 初始化分词器
      dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #B 创建GPTDatasetV1类
      dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,  #C drop_last=True会在最后一批次小于指定的batch_size时丢弃该批次，以防止训练期间的损失峰值
            num_workers=0         #D 用于预处理的CPU进程数量
      )

      return dataloader
with open("The_Verdict.txt", "r", encoding="utf-8") as f: 
      raw_text = f.read()

#dataloader = create_dataloader_v1(
#      raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
#data_iter = iter(dataloader)     #获取下一个数据条目
#first_batch = next(data_iter)
#print(first_batch)
#second_batch = next(data_iter)
#print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# 2.7构建词向量
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))

# 2.8 位置编码
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(
      raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

