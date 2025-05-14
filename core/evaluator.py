from typing import Dict, Any
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel
from tqdm import tqdm

from utils.model_utils import init_custom_smolvlm
from dataset_utils.dataset_formatter import format_data, gather_images_from_message
from core.utils.generate_text import generate_text_from_sample
from metrics.cider import download_and_prepare, compute


def eval_data_collate_fn(examples, processor):
    texts = []
    images = []
    gt = []
    for example in examples:
        image_inputs = gather_images_from_message(example)
        text = processor.apply_chat_template(
            example[1:2], add_generation_prompt=True, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)
        gt.append(example[2]["content"][0]["text"])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return batch, gt


def infer_model(model, processor, eval_dataloader):
    all_preds = []
    all_gt = []

    for batch, gt in tqdm(eval_dataloader):
        pred = generate_text_from_sample(model, processor, batch.to(model.device))
        all_preds.extend(pred)
        all_gt.extend(gt)

    return all_preds, all_gt


def evaluate(config: Dict[str, Any]) -> None:
    """ main eval function"""

    model, processor = init_custom_smolvlm(config['model']['model_id'])
    model = model.eval()

    eval_dataset = load_dataset(config["dataset"]["id"], cache_dir="/media/vishal/datasets/hf_cache/")
    eval_dataset = eval_dataset["val"]
    test_dataset = [format_data(sample) for sample in eval_dataset]

    # Create dataloader
    collate = partial(eval_data_collate_fn, processor=processor)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate, num_workers=4, pin_memory=True)

    # prepare cider metric
    tokenizer_path = download_and_prepare(save_path="/media/vishal/workspace/projects/AlignVLM")

    # Generate captions
    pred, gt = infer_model(model, processor, eval_dataloader)

    if config['model']['lora_adapter_path']:
        model = PeftModel.from_pretrained(config['model']['lora_adapter_path'])

    cider_score = compute(pred, gt, tokenizer_path)



