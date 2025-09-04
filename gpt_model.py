import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from layer_norm import LayerNorm
"""
Takes in *dict* configuration data for llm and creates a GPT Model for running

Methods:
    :forward: Core logic of the GPT Model; 
    1.takes in input tokens and creates vectors for each of them
    2.creates positional arguments for those tokens
    3.Applies transformer block(attention ect..) to that data 
    4.Normalizes the data
    5.Gets the postional data
    6.Get the logits for final output
        :params: input tokens(original sentence/input in tokenized form)

    :init: Gets data from the input config parameter and links it to  
"""
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Embedding Layers for tokens + Position Embedding
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout Rate
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits