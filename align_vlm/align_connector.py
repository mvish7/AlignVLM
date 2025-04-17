"""
a connector/projector module proposed in - https://arxiv.org/abs/2502.01341
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# todo: use weights from lm_head for W2, create a method to patch weights of lm_head, as you must init the llm first.

class ALIGNModule(nn.Module):
    def __init__(self, config):
        super(ALIGNModule, self).__init__()
        # vision_dim: int = 768, llm_dim: int = 960, vocab_size: int = 49280
        # config.vision_config.hidden_size
# config.text_config.vocab_size
        # config.text_config.hidden_size
        self.scale_factor = config.scale_factor

        self.W1 = nn.Linear(config.vision_config.hidden_size*(config.scale_factor ** 2), config.text_config.hidden_size)
        self.W2 = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def pixel_shuffle(self, x, scale_factor=2):
        """
        Augmentation to carry out patch/pixels merging, reduces sequence length as well.
        Carried over from SmolVLMConnector class in modelling_smolvlm.py
        """
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))

        return x

    def forward(self, vision_feats, llm_token_embed):
        """
        vision_feats: (B, T*N, d)
        tokenized_llm_input: (B, L)
        """
        # Step0: apply pixel shuffle
        vision_feats = self.pixel_shuffle(vision_feats, self.scale_factor)
        #todo: vision_feats should be (13, 64, 12288)

        # Step 1: Linear project to LLM dimension
        F_proj = self.W1(vision_feats)  # (B, T*N, D)

        # Step 2: Normalize + project to vocab logits
        F_proj = F.layer_norm(F_proj, F_proj.shape[-1:])
        vocab_logits = self.W2(F_proj)  # (B, T*N, V)

        # Step 3: Softmax over vocabulary dimension
        P_vocab = F.softmax(vocab_logits, dim=-1)  # (B, T*N, V)

        # Step 4: Weighted sum over LLM embeddings
        F_align = torch.matmul(P_vocab, llm_token_embed)  # (B, T*N, D)
        # todo: F_align has shape of (13, 1024, 960)

        #todo: F_align equivalent (projector output in smolvlm) is (13, 64, 960)


        # todo: E_text_x should be (1, 876, 960)
        # # Step 5: Token embeddings from input x
        # E_text_x = self.llm_token_embed[tokenized_llm_input]  # (B, L, D)
        #
        # # Step 6: Concatenate F_align and E_text(x)
        # H_input = torch.cat([F_align, E_text_x], dim=1)  # (B, T*N + L, D)
        # todo: H_Input: should be (1, 876, 960)

        # todo: if you merge_inputs using alignVLM's approach you need to override the inputs merger function from smolVLM
        return F_align

        # todo: compare this with original smolVLM inputs_merger and adapt.
