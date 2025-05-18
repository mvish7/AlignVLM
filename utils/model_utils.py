import torch

from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.models.smolvlm import modeling_smolvlm
from align_vlm.align_connector import ALIGNModule
from align_vlm.customized_smolvlm_model import CustomSmolVLMModel


def init_smolvlm(model_id, use_align_connector=True, is_train=True):
    if use_align_connector:
        modeling_smolvlm.SmolVLMModel = CustomSmolVLMModel

    if is_train:
        processor = AutoProcessor.from_pretrained(model_id)  # padding_side="left"
    else:
        processor = AutoProcessor.from_pretrained(model_id, padding_side="left")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        _attn_implementation="eager").to("cuda")

    return model, processor


def freeze_for_stage_1_training(model):
    # Freeze the parameters of the vision tower
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False

    # Freeze the parameters of the llm
    for name, param in model.named_parameters():
        if "text_model" in name:
            param.requires_grad = False

    # verify that necessary parameters are frozen
    for name, param in model.named_parameters():
        if param.requires_grad == True and ("vision_model" not in name or "text_model" not in name):
            print(name)

    return model


def freeze_for_stage_2_training(model):
    # Freeze the parameters of the vision tower
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False

    # verify that necessary parameters are frozen
    for name, param in model.named_parameters():
        if param.requires_grad == True and ("vision_model" not in name):
            print(name)

    return model
