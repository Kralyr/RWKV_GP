import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import argparse

# Time Mix Layer
class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, arg):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = arg.ctx_len
        self.n_head = arg.D_n_head
        self.head_size = arg.D_n_attn // self.n_head

        with torch.no_grad():
            ww = torch.ones(self.n_head, self.ctx_len)
            curve = torch.tensor([-(self.ctx_len - 1 - i) for i in range(self.ctx_len)])
            for h in range(self.n_head):
                if h < self.n_head - 1:
                    decay_speed = math.pow(self.ctx_len, -(h+1)/(self.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)

        self.time_w = nn.Parameter(ww)
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, self.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, self.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(self.ctx_len, 1))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(arg.D_n_embd, arg.D_n_attn)
        self.value = nn.Linear(arg.D_n_embd, arg.D_n_attn)
        self.receptance = nn.Linear(arg.D_n_embd, arg.D_n_attn)
        self.output = nn.Linear(arg.D_n_attn, arg.D_n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:]
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)
        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)
        rwkv = torch.sigmoid(r) * wkv / sum_k
        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]

# Channel Mix Layer
class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id, arg):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        hidden_sz = 5 * arg.D_n_ffn // 2
        self.key = nn.Linear(arg.D_n_embd, hidden_sz)
        self.value = nn.Linear(arg.D_n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, arg.D_n_embd)
        self.receptance = nn.Linear(arg.D_n_embd, arg.D_n_embd)

    def forward(self, x):
        B, T, C = x.size()
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)
        rwkv = torch.sigmoid(r) * wkv
        return rwkv

# Main Model Block
class Block(nn.Module):
    def __init__(self, layer_id, arg):
        super().__init__()

        self.ln1 = nn.LayerNorm(arg.D_n_embd)
        self.ln2 = nn.LayerNorm(arg.D_n_embd)
        self.attn = RWKV_TimeMix(layer_id, arg)
        self.mlp = RWKV_ChannelMix(layer_id, arg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Final RWKV Model
class RWKV(nn.Module): #RWKV(self.config,self.n_channels)
    def __init__(self, arg, n_channels):
        super().__init__()
        # print(arg.ctx_len)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, arg.D_n_embd),
        )

        self.blocks = nn.Sequential(*[Block(i, arg) for i in range(arg.D_n_layer)])
        self.ln_f = nn.LayerNorm(arg.D_n_embd)

    def forward(self, x):
        x = self.fc(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

# # Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="RWKV Time Series Model")
    parser.add_argument("--D_ctx_len", type=int, default=256, help="Context length (default: 256)")

    parser.add_argument("--D_n_layer", type=int, default=8, help="Number of layers (default: 8)")
    parser.add_argument("--D_n_head", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--D_n_embd", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--D_n_attn", type=int, default=512, help="Attention dimension (default: 512)")
    parser.add_argument("--D_n_ffn", type=int, default=512, help="Feed-forward network dimension (default: 512)")




    return parser.parse_args()

# # Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    arg = parse_args()
    # print(arg.attrlen)
    model = RWKV(arg)

    # Print the model configuration
    print("Model parameters:", arg)
    
    # Example dummy input
    dummy_input = torch.randn(16, 256, 37)  # Batch size: 16, Context length: 256, Feature size: 14
    output = model(dummy_input)
    print("Model output shape:", output.shape)
