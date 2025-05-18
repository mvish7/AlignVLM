from PIL import Image


def prep_sample_for_generation(processor, sample, device, model_dtype):
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
    ).to(device).to(model_dtype)  # Move inputs to the specified device