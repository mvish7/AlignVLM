import random
from typing import Dict, Any, List, Tuple
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm

from utils.model_utils import init_smolvlm
from dataset_utils.dataset_formatter import (format_data,
                                             gather_images_from_message)
from core.utils.generate_text import generate_text_from_sample
from metrics.cider import download_and_prepare, compute


def eval_data_collate_fn(examples: List[Dict],
                         processor: Any) -> Tuple[Dict, List[str]]:
    """Collate function for evaluation dataloader.
    
    Args:
        examples: List of examples to collate
        processor: Model processor for text and image processing
        
    Returns:
        Tuple containing processed batch and ground truth texts
    """
    texts = []
    images = []
    gt = []

    for example in examples:
        image_inputs = gather_images_from_message(example)
        text = processor.apply_chat_template(example[1:2],
                                             add_generation_prompt=True,
                                             tokenize=False)
        texts.append(text.strip())
        images.append(image_inputs)
        gt.append(example[2]["content"][0]["text"])

    batch = processor(text=texts,
                      images=images,
                      return_tensors="pt",
                      padding=True)
    return batch, gt


def infer_model(model: Any, processor: Any,
                eval_dataloader: DataLoader) -> Tuple[List[str], List[str]]:
    """Run inference on the evaluation dataset.
    
    Args:
        model: Model to run inference with
        processor: Model processor
        eval_dataloader: DataLoader containing evaluation data
        
    Returns:
        Tuple containing predictions and ground truth texts
    """
    all_preds = []
    all_gt = []

    for batch, gt in tqdm(eval_dataloader):
        pred = generate_text_from_sample(model,
                                         processor,
                                         batch.to(model.device),
                                         processed_sample=True)
        all_preds.extend(pred)
        all_gt.extend(gt)

    return all_preds, all_gt


def evaluate(config: Dict[str, Any]) -> None:
    """Main evaluation function.
    
    Args:
        config: Configuration dictionary containing model, dataset and 
               evaluation settings
    """
    # Initialize model
    model, processor = init_smolvlm(
        config['model']['model_id'],
        use_align_connector=config['model']['use_align_connector'],
        is_train=config['model']['is_train'])

    if config['model']['lora_adapter_path']:
        model = PeftModel.from_pretrained(model,
                                          config['model']['lora_adapter_path'])
    model = model.eval()

    # Load and prepare dataset
    eval_dataset = load_dataset(config["dataset"]["id"],
                                cache_dir=config["dataset"]["cache_dir"])
    eval_dataset = eval_dataset["val"]

    if config['dataset']['few_shot']['enabled']:
        eval_dataset2 = []
        for sample in eval_dataset:
            few_shot_ex = random.choices(
                eval_dataset, k=config['dataset']['few_shot']['num_examples'])
            eval_dataset2.append(
                format_data(sample, few_shot_ex, test_set=True))
    else:
        eval_dataset2 = [
            format_data(sample, test_set=True) for sample in eval_dataset
        ]

    # Create dataloader
    collate = partial(eval_data_collate_fn, processor=processor)
    eval_dataloader = DataLoader(eval_dataset2,
                                 batch_size=config['eval']['batch_size'],
                                 collate_fn=collate,
                                 num_workers=config['eval']['num_workers'],
                                 pin_memory=config['eval']['pin_memory'])

    # Prepare CIDEr metric
    tokenizer_path = download_and_prepare(
        save_path=config['eval']['cider_tokenizer_path'])

    # Generate captions and compute score
    pred, gt = infer_model(model, processor, eval_dataloader)
    cider_score = compute(pred, gt, tokenizer_path)

    print(f"CIDEr score is - {cider_score}")
