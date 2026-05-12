# 2.2 预处理文本数据
with open("The_Verdict.txt", "r", encoding="utf-8") as f:  # 打开文本文件，读取内容
	raw_text = f.read() 
	print("Total number of character:", len(raw_text)) 
	print(raw_text[:99])

import re # 导入正则表达式模块
text = "Hello, world. Is this-- a test."
result = re.split(r'(\s)', text) #拆分成「单词 + 标点符号」的干净列表,把空格、多余符号都处理掉。
print(result)
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)  #拆分成「单词 + 标点符号」的干净列表，保留标点符号
print(result)
result = [item for item in result if item.strip()] #去掉空字符串和纯空格的字符串
print(result)

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 拆分成「单词 + 标点符号」的干净列表，保留标点符号
result = [item.strip() for item in result if item.strip()] # 去掉空字符串和纯空格的字符串
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

# 2.3  将 tokens 转换为token IDs
all_words = sorted(set(preprocessed)) 
# sorted()函数对可迭代对象进行排序，返回一个新的列表。
# set()函数创建一个无序不重复元素集，常用来去除重复元素。
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)} # enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标，常用在for循环中。
for i, item in enumerate(vocab.items()): # items()方法返回一个包含字典键值对的视图对象，可以使用list()函数将其转换为列表。
    print(item)
    if i > 50:
        break 

# 分词器V1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab                                                   #A
        self.int_to_str = {i:s for s,i in vocab.items()}                          #B

    def encode(self, text):                                                       #C
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):                                                        #D
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)                           #E
        return text

#A 将词汇表作为类属性存储，以方便在 encode 和 decode 方法中访问
#B 创建一个反向词汇表，将token ID 映射回原始的文本token
#C 将输入文本转换为token ID
#D 将token ID 还原为文本
#E 在指定的标点符号前去掉空格

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

#text = "Hello, do you like tea?"
#print(tokenizer.encode(text)) # KeyError: 'Hello' 它不包含在词汇中

#2.4添加特殊上下文token
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# 分词器V2
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int                    #A
                    else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)                    #B
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "  " + "  ".join((text1, text2)) + "  "
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))




