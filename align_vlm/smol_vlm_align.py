from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.smolvlm import modeling_smolvlm
import torch
import torch.nn as nn


class SmolVLMSimpleMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        input_size = 768 * (4**2)
        output_size = 960
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class CustomConnector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.scale_factor = 4
        self.modality_projection = SmolVLMSimpleMLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor),
                   embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz,
                      int(width / scale_factor), int(height / scale_factor),
                      embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)),
                      embed_dim * (scale_factor**2))

        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states,
                                                 self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


if __name__ == "__main__":
    messages = [
        {
            "role":
            "user",
            "content": [
                {
                    "type":
                    "image",
                    "path":
                    "/media/vishal/datasets/hypersim/downloads/ai_001_002/images/scene_cam_00_final_preview/frame.0041.tonemap.jpg"
                },
                {
                    "type": "text",
                    "text": "Can you describe this image?"
                },
            ]
        },
    ]

    # ------------------------------------------------------------------------------------------------------------------

    model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    modeling_smolvlm.SmolVLMConnector = CustomConnector

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        _attn_implementation="eager").to("cuda")

    # ------------------------------------------------------------------------------------------------------------------

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs,
                                   do_sample=False,
                                   max_new_tokens=512)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])

    # --------------------------
    # weights of embed layer
    # model.model.text_model.embed_tokens.weight
    # weights of LM head --
    # model.lm_head.weight
