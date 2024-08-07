import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import DualDecoderTransformer

from torch.nn.utils.rnn import pad_sequence


class TrainArgs:
    # 超参数
    vocab_size = 10000  # 词汇表大小
    d_model = 512  # 模型维度
    nhead = 8  # 注意力头数
    num_layers = 6  # 解码器层数
    dim_feedforward = 2048  # 前馈网络维度
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001


args = TrainArgs()
# 设置设备
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDataset(Dataset):
    def __init__(self, num_samples, max_seq_length):
        self.data = [torch.randint(1, args.vocab_size, (torch.randint(10, max_seq_length, (1,)).item(),))
                     for _ in range(num_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    # 对序列进行填充
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    # 创建掩码
    mask = torch.triu(torch.ones(padded_batch.size(1), padded_batch.size(1)), diagonal=1).bool()

    return padded_batch, mask


# 创建数据集和数据加载器
train_dataset = SimpleDataset(10000, 50)  # 10000个样本，每个长度为50
_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# 初始化模型
_model = DualDecoderTransformer(args.vocab_size, args.d_model, args.nhead, args.num_layers, args.dim_feedforward).to(
    _device)

# 定义损失函数和优化器
_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充标记
_optimizer = optim.Adam(_model.parameters(), lr=args.learning_rate)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch, mask in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        mask = mask.to(device)

        # 前向传播
        output, direction_logits = model(batch, mask)

        # 计算损失
        loss = criterion(output.view(-1, args.vocab_size), batch.view(-1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# 训练循环
for epoch in range(args.num_epochs):
    avg_loss = train_epoch(_model, _dataloader, _criterion, _optimizer, _device)
    print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}")

# 保存模型
torch.save(_model.state_dict(), "dual_decoder_transformer.pth")
