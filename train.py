import torch
from torch import nn

from model import DualDecoderTransformer

# 初始化模型和训练
vocab_size = 10000  # 示例词汇表大小
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048

model = DualDecoderTransformer(vocab_size, d_model, nhead, num_layers, dim_feedforward)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# 假设我们有一个 train_dataloader
# train(model, optimizer, criterion, train_dataloader, num_epochs=10)