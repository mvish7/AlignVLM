from PIL import Image
from  datasets import load_from_disk
from dataset_utils.dataset_formatter import format_data
from utils.model_utils import init_smolvlm

from core.utils.generate_text import generate_text_from_sample


if __name__ == "__main__":
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    dataset_path = "/media/vishal/datasets/pixmo/metadata/"
    lora_adapter_path = "/media/vishal/workspace/projects/gemma3_4b/align_vlm/checkpoints/pixmo_100k"

    model, processor = init_smolvlm(model_id, use_align_connector=True, is_train=False)
    model.load_adapter(lora_adapter_path)

    dataset = load_from_disk(dataset_path)
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

    train_dataset = dataset_dict["train"].shuffle()
    test_dataset = dataset_dict["test"].shuffle()

    train_dataset = [format_data(sample) for sample in train_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    sample = test_dataset[100]
    print("-------------------------------------------------------")
    print(f"image -- {sample[1]['content'][0]['image']}")
    print(f"GT caption is -- {sample[2]['content'][0]['text']}")
    print("-------------------------------------------------------")
    generated_caption = generate_text_from_sample(model, processor, sample)
    print(generated_caption)