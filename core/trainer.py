"""Training script for AlignVLM stage 2."""

from typing import Dict, Any
from PIL import Image
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
from datasets import load_from_disk

from dataset_utils.dataset_formatter import format_data
from utils.model_utils import init_smolvlm, freeze_for_stage_2_training


def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Create LoRA configuration from config dict."""
    # targetting attention and mlp for lora
    llm_target_modules = []
    for i in range(config['model']['num_transformer_layers']):
        llm_target_modules.extend([
            f"text_model.layers.{i}.self_attn.q_proj",
            f"text_model.layers.{i}.self_attn.k_proj",
            f"text_model.layers.{i}.self_attn.v_proj",
            f"text_model.layers.{i}.self_attn.o_proj",
            f"text_model.layers.{i}.mlp.gate_proj",
            f"text_model.layers.{i}.mlp.up_proj",
            f"text_model.layers.{i}.mlp.down_proj"
        ])

    # keeping following layers fully trainable
    norm_layers_to_save = []
    for i in range(34):
        norm_layers_to_save.extend([
            f"text_model.layers.{i}.input_layernorm",
            f"text_model.layers.{i}.post_attention_layernorm"
        ])

    all_modules_to_save = ["lm_head", "embed_tokens", "connector"
                           ] + norm_layers_to_save

    return LoraConfig(
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        r=config['lora']['r'],
        use_rslora=config['lora']['use_rslora'],
        bias=config['lora']['bias'],
        target_modules=llm_target_modules,
        task_type=config['lora']['task_type'],
        modules_to_save=all_modules_to_save,
    )


def create_data_collator(processor, image_token_id):
    """Create data collator function for training."""

    def collate_fn(examples):
        # applying chat template from smol_vlm
        texts = [
            processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]
        # gathering images
        image_inputs = []
        for example in examples:
            image_path = example[1]["content"][0]["image"]
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_inputs.append([image])
        # processing image and text
        batch = processor(text=texts,
                          images=image_inputs,
                          return_tensors="pt",
                          padding=True)
        # prep labels - mask pad and image tokens
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def prepare_datasets(config: Dict[str, Any]) -> tuple:
    """Prepare train and test datasets."""
    dataset = load_from_disk(config['dataset']['path'])
    dataset_dict = dataset.train_test_split(
        test_size=config['dataset']['test_size'],
        seed=config['dataset']['seed'],
        shuffle=True)

    train_dataset = dataset_dict["train"].shuffle()
    test_dataset = dataset_dict["test"].shuffle()

    train_dataset = [format_data(sample) for sample in train_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    return train_dataset, test_dataset


def train_model(config: Dict[str, Any]) -> None:
    """Main training function."""
    # Initialize model and processor
    model, processor = init_smolvlm(config['model']['model_id'], use_align_connector=True)
    model = freeze_for_stage_2_training(model)

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]

    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(config)

    # Setup LoRA
    if not config['model']['resume_from_ckpt']:
        peft_config = create_lora_config(config)
    else:
        model = PeftModel.from_pretrained(model,
                                          config['model']['lora_adapter_path'],
                                          is_trainable=True)

    # Setup training arguments
    training_args = SFTConfig(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        optim=config['training']['optim'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        learning_rate=config['training']['learning_rate'],
        bf16=config['training']['bf16'],
        report_to=config['training']['report_to'],
        eval_strategy=config['training']['eval_strategy'],
        eval_accumulation_steps=config['training']['eval_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        per_device_train_batch_size=config['training']
        ['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']
        ['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']
        ['gradient_accumulation_steps'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        gradient_checkpointing_kwargs=config['training']
        ['gradient_checkpointing_kwargs'],
        weight_decay=config['training']['weight_decay'],
        use_cpu=config['training']['use_cpu'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        torch_empty_cache_steps=config['training']['torch_empty_cache_steps'],
        dataset_kwargs=config['training']['dataset_kwargs'],
    )

    # Create data collator
    collate_fn = create_data_collator(processor, image_token_id)

    # Initialize trainer
    if not config['model']['resume_from_ckpt']:
        model_trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=collate_fn,
            eval_dataset=test_dataset,
        )
    else:
        model_trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
            data_collator=collate_fn,
            eval_dataset=test_dataset,
        )

    # Train and save model
    model_trainer.train()
    model_trainer.save_model()
