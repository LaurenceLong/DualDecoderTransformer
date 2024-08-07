import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, attn_mask=mask)[0]
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)


class DirectionSelector(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, l_hidden, r_hidden):
        l_attn = self.self_attn(l_hidden, l_hidden, l_hidden)[0]
        r_attn = self.self_attn(r_hidden, r_hidden, r_hidden)[0]

        l_cross = self.cross_attn(l_attn, r_attn, r_attn)[0]
        r_cross = self.cross_attn(r_attn, l_attn, l_attn)[0]

        l_pooled = self.pool(l_cross.transpose(1, 2)).squeeze(2)
        r_pooled = self.pool(r_cross.transpose(1, 2)).squeeze(2)

        combined = torch.cat([l_pooled, r_pooled], dim=-1)
        return self.mlp(combined)


class DualDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.l_decoder = Decoder(vocab_size, d_model, nhead, num_layers, dim_feedforward)
        self.r_decoder = Decoder(vocab_size, d_model, nhead, num_layers, dim_feedforward)
        self.selector = DirectionSelector(d_model, nhead)

    def forward(self, x, l_mask, r_mask):
        l_output = self.l_decoder(x, l_mask)
        r_output = self.r_decoder(x, r_mask)
        direction = self.selector(l_output, r_output)
        return l_output, r_output, direction


# 辅助函数
def create_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


def train_step(model, optimizer, criterion, input_ids, target_ids):
    optimizer.zero_grad()

    batch_size, seq_len = input_ids.shape
    l_mask = create_mask(seq_len).to(input_ids.device)
    r_mask = l_mask.flip(dims=[0, 1])

    l_output, r_output, direction = model(input_ids, l_mask, r_mask)

    l_loss = criterion(l_output.view(-1, l_output.size(-1)), target_ids.view(-1))
    r_loss = criterion(r_output.view(-1, r_output.size(-1)), target_ids.flip(dims=[1]).view(-1))

    # 假设我们有真实的方向标签
    direction_loss = F.cross_entropy(direction.view(-1, 2), torch.randint(0, 2, (batch_size,)).to(input_ids.device))

    total_loss = l_loss + r_loss + direction_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# 训练循环
def train(model, optimizer, criterion, train_dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, target_ids = batch
            loss = train_step(model, optimizer, criterion, input_ids, target_ids)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")


# 生成函数
def generate(model, start_tokens, max_length, temperature=1.0):
    model.eval()
    current_output = start_tokens.clone()
    left_generated = 0
    right_generated = 0

    for _ in range(max_length):
        l_mask = create_mask(current_output.size(1)).to(start_tokens.device)
        r_mask = l_mask.flip(dims=[0, 1])

        with torch.no_grad():
            l_output, r_output, direction = model(current_output, l_mask, r_mask)

        direction_prob = F.softmax(direction, dim=-1)
        chosen_direction = torch.argmax(direction_prob, dim=-1).item()

        if chosen_direction == 0:  # Left
            next_token_logits = l_output[:, 0, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            current_output = torch.cat([next_token, current_output], dim=1)
            left_generated += 1
        else:  # Right
            next_token_logits = r_output[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            current_output = torch.cat([current_output, next_token], dim=1)
            right_generated += 1

        # 可选：检查是否达到最大长度
        if left_generated + right_generated + start_tokens.size(1) >= max_length:
            break

    return current_output
