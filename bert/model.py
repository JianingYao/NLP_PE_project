# model.py
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import math

class CustomBertModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pos_embedding_type='sinusoidal'):
        super(CustomBertModel, self).__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        
        # Replace BERT's default position embeddings
        if pos_embedding_type == 'sinusoidal':
            self.bert.embeddings.position_embeddings = nn.Embedding.from_pretrained(
                self.sinusoidal_embeddings(config.max_position_embeddings, config.hidden_size), freeze=True)
        elif pos_embedding_type == 'alibi':
            self.bert.embeddings.position_embeddings = AlibiPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        elif pos_embedding_type == 'rope':
            self.bert.embeddings.position_embeddings = RoPEPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

    def sinusoidal_embeddings(self, num_positions, hidden_dim):
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        sinusoidal_embedding = torch.zeros((num_positions, hidden_dim))
        sinusoidal_embedding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_embedding[:, 1::2] = torch.cos(position * div_term)
        return sinusoidal_embedding

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.classifier(outputs.pooler_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else logits

class AlibiPositionalEmbedding(nn.Module):
    def __init__(self, num_positions, hidden_dim):
        super().__init__()
        self.position_bias = nn.Parameter(torch.arange(num_positions).float().unsqueeze(0).repeat(hidden_dim // 2, 1).T)
    
    def forward(self, x):
        # Typically, Alibi adjustments would be done inside the attention mechanism
        # For a simplified version, we can directly add a learned bias based on position
        return x + self.position_bias[:x.size(1)].to(x.device)

class RoPEPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
    def forward(self, x):
        seq_len, batch_size, embed_dim = x.shape
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, self.inv_freq)
        sin_inp = sinusoid_inp.sin()
        cos_inp = sinusoid_inp.cos()
        # Apply Rotary Position Encoding
        q = x.reshape(seq_len, batch_size, embed_dim // 2, 2)  # Split into (real, imaginary) parts
        q_rot = torch.stack((-q[..., 1], q[..., 0]), dim=-1)  # Rotate by 90 degrees
        q_transformed = q * cos_inp.unsqueeze(-1) + q_rot * sin_inp.unsqueeze(-1)
        return q_transformed.reshape(seq_len, batch_size, embed_dim)
