import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig


class IntelligentBoxSplitter:
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def split_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        boxes = []
        current_box = []

        for token in tokens:
            if len(current_box) < self.max_length:
                current_box.append(token)
            else:
                boxes.append(current_box)
                current_box = [token]

        if current_box:
            boxes.append(current_box)

        return boxes


class LDecoder(nn.Module):
    def __init__(self, config):
        super(LDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads),
            num_layers=config.num_hidden_layers)
        self.position_encoding = nn.Parameter(torch.randn(1, config.max_position_embeddings, config.hidden_size))

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return self.transformer_decoder(x, x)


class RDecoder(nn.Module):
    def __init__(self, config):
        super(RDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads),
            num_layers=config.num_hidden_layers)
        self.position_encoding = nn.Parameter(torch.randn(1, config.max_position_embeddings, config.hidden_size))

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return self.transformer_decoder(x, x)


class DirectionSelector(nn.Module):
    def __init__(self, hidden_size):
        super(DirectionSelector, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2))

    def forward(self, l_hidden, r_hidden):
        l_attn, _ = self.self_attention(l_hidden, l_hidden, l_hidden)
        r_attn, _ = self.self_attention(r_hidden, r_hidden, r_hidden)
        cross_attn, _ = self.cross_attention(l_attn, r_attn, r_attn)

        pooled = self.pooling(cross_attn.transpose(1, 2)).squeeze(-1)
        decision_logits = self.mlp(pooled)

        return torch.softmax(decision_logits, dim=-1)


class BiDirectionalTextGenerator(nn.Module):
    def __init__(self, config):
        super(BiDirectionalTextGenerator, self).__init__()
        self.box_splitter = IntelligentBoxSplitter()
        self.l_decoder = LDecoder(config)
        self.r_decoder = RDecoder(config)
        self.direction_selector = DirectionSelector(config.hidden_size)

    def forward(self, text):
        boxes = self.box_splitter.split_text(text)
        for box in boxes:
            l_output = self.l_decoder(box)
            r_output = self.r_decoder(box)

            direction_prob = self.direction_selector(l_output, r_output)
            chosen_decoder = self.l_decoder if direction_prob[0] > direction_prob[1] else self.r_decoder

            # Continue generating with chosen_decoder
            # Additional logic to complete the generation


# Example use case
config = BertConfig()
text_generator = BiDirectionalTextGenerator(config)
text = "Your input text goes here."
output = text_generator(text)