"""
a connector/projector module as explained in - https://arxiv.org/abs/2502.01341
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# todo: use weights from lm_head for W2, create a method to patch weights of lm_head, as you must init the llm first.

class ALIGNModule(nn.Module):
    def __init__(self, vision_dim, llm_dim, vocab_size, llm_token_embed):
        super(ALIGNModule, self).__init__()
        self.W1 = nn.Linear(vision_dim, llm_dim)
        self.W2 = nn.Linear(llm_dim, vocab_size, bias=False)
        self.llm_token_embed = llm_token_embed  # E_text: (V, D)

    def forward(self, vision_feats):
        """
        vision_feats: (B, T*N, d)
        tokenized_llm_input: (B, L)
        """
        # Step 1: Linear project to LLM dimension
        F_proj = self.W1(vision_feats)  # (B, T*N, D)

        # Step 2: Normalize + project to vocab logits
        F_proj_norm = F.layer_norm(F_proj, F_proj.shape[-1:])
        vocab_logits = self.W2(F_proj_norm)  # (B, T*N, V)

        # Step 3: Softmax over vocabulary dimension
        P_vocab = F.softmax(vocab_logits, dim=-1)  # (B, T*N, V)

        # Step 4: Weighted sum over LLM embeddings
        F_align = torch.matmul(P_vocab, self.llm_token_embed)  # (B, T*N, D)

        # # Step 5: Token embeddings from input x
        # E_text_x = self.llm_token_embed[tokenized_llm_input]  # (B, L, D)
        #
        # # Step 6: Concatenate F_align and E_text(x)
        # H_input = torch.cat([F_align, E_text_x], dim=1)  # (B, T*N + L, D)

        return H_input
