from PIL import Image
from  datasets import load_from_disk
from align_vlm.dataset_utils.dataset_formatter import format_data
from align_vlm.utils.model_utils import init_custom_smolvlm


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2], add_generation_prompt=True  # Use the sample without the system message
    )

    image_inputs = []
    image_path = sample[1]["content"][0]["image"]
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        # text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device).to(model.dtype)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


if __name__ == "__main__":
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    dataset_path = "/media/vishal/datasets/pixmo/metadata/"
    lora_adapter_path = "/media/vishal/workspace/projects/gemma3_4b/align_vlm/checkpoints/pixmo_100k"

    model, processor = init_custom_smolvlm(model_id)
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