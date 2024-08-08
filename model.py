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
        self.last_attn_scores = None

    def forward(self, x, mask):
        x2 = self.norm1(x)
        attn_output, self.last_attn_scores = self.self_attn(x2, x2, x2, attn_mask=mask)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x

    def get_attention_scores(self):
        return self.last_attn_scores


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

    def get_last_attention_scores(self):
        return self.layers[-1].get_attention_scores()


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


class BalancedDirectionSelector(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention_projection = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, l_state, r_state, context, l_attn_scores, r_attn_scores):
        # l_state, r_state: [batch_size, seq_len, d_model]
        # context: [batch_size, seq_len, d_model]
        # l_attn_scores, r_attn_scores: [batch_size, nhead, seq_len, seq_len]

        # 使用最后一个时间步的状态作为查询
        l_query = l_state[:, -1:, :]
        r_query = r_state[:, -1:, :]
        combined_query = torch.cat([l_query, r_query], dim=1)

        # 平均多头注意力分数
        l_attn_scores = l_attn_scores.mean(dim=1)  # [batch_size, seq_len, seq_len]
        r_attn_scores = r_attn_scores.mean(dim=1)  # [batch_size, seq_len, seq_len]

        # 使用注意力分数加权上下文
        l_context_vector = torch.bmm(l_attn_scores, context)[:, -1, :]  # [batch_size, d_model]
        r_context_vector = torch.bmm(r_attn_scores, context)[:, -1, :]  # [batch_size, d_model]

        # 结合两个方向的上下文信息
        context_vector = self.attention_projection(l_context_vector + r_context_vector)

        # 结合两个方向的最后状态和上下文信息
        l_last = l_state[:, -1, :]  # [batch_size, d_model]
        r_last = r_state[:, -1, :]  # [batch_size, d_model]
        combined = torch.cat([l_last, r_last, context_vector], dim=1)

        # 通过MLP得到最终的选择概率
        logits = self.mlp(combined)
        return logits


class DualDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.l_decoder = Decoder(vocab_size, d_model, nhead, num_layers, dim_feedforward)
        self.r_decoder = Decoder(vocab_size, d_model, nhead, num_layers, dim_feedforward)
        self.selector = BalancedDirectionSelector(d_model, nhead)
        self.d_model = d_model

    def forward(self, x, mask):
        # 运行左右两个解码器
        l_output = self.l_decoder(x, mask)
        r_output = self.r_decoder(x, mask)

        # 获取最后一层的注意力分数
        l_attn_scores = self.l_decoder.get_last_attention_scores()
        r_attn_scores = self.r_decoder.get_last_attention_scores()

        # 使用 selector 选择方向
        context = (l_output + r_output) / 2  # 简单地将两个输出平均作为上下文
        direction_logits = self.selector(l_output, r_output, context, l_attn_scores, r_attn_scores)

        # 根据 logits 选择最终输出
        direction_probs = F.softmax(direction_logits, dim=-1)
        final_output = direction_probs[:, 0].unsqueeze(1).unsqueeze(2) * l_output + \
                       direction_probs[:, 1].unsqueeze(1).unsqueeze(2) * r_output

        return final_output, direction_logits


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
