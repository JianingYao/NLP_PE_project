import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
import inspect
from torch.autograd import Variable

device="cuda:0"
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) +get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def alibi_bias(size):
    result=[]
    list0=[-i for i in range(size)]
    result.append(list0)
    for j in range(1,size):
        list=[0 for i in range(j)]+[list0[i] for i in range(size-j)]
        result.append(list)
    result=np.array(result)
    result=torch.from_numpy(result).T.to(torch.float16).to(device)
    return result



def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None,num_head=12) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype,device=device)
    slopes = torch.Tensor(get_slopes(num_head)).to(device)
    slopes=slopes.unsqueeze(0).unsqueeze(1)
    alibi=alibi_bias(L).unsqueeze(-1)

    alibi_3d=alibi @ slopes

    alibi_3d=alibi_3d.permute(2,0,1)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool,device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += alibi_3d
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class MHA(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop,maxpos=5000):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.ln = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model)
        self.slopes = torch.Tensor(get_slopes(num_heads))
        # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
        # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        # This works because the softmax operation is invariant to translation, and our bias functions are always linear.
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(
            num_heads, -1, -1)
        self.alibi = self.alibi.view(num_heads, 1, maxpos)
        self.alibi = self.alibi.repeat(16, 1, 1)  # batch_size, 1, 1

    def forward(self, x):
        B, T, C = x.size()
        x_qkv = self.ln(x)
        q, k, v = x_qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_pdrop if self.training else 0,
                                         is_causal=True,num_head=self.num_heads)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.ln1 = nn.Linear(d_model, dff)
        self.ln2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, num_heads, attn_pdrop, resid_pdrop)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)

    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return self.dropout(x)  # size = [batch, L, d_model]


class GPT2_alibi(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, embed_pdrop, num_heads, dff, attn_pdrop, resid_pdrop, dropout,
                 num_layer):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, sparse=False)
        # self.pos_embed = nn.Embedding(block_size, d_model, sparse=False)
        self.pos_embed = PositionalEncoding(d_model, dropout)
        self.dropout_embed = nn.Dropout(embed_pdrop)
        # self.blocks = [Block(d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout) for _ in range(num_layer)]
        self.blocks = nn.ModuleList(
            [Block(d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout) for _ in range(num_layer)])
        self.num_layer = num_layer
        self.block_size = block_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.layernorm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()

        x = self.token_embed(x)
        pos_embed = self.pos_embed(x)

        #x = x + pos_embed
        x = self.dropout_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, block_size=512):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx