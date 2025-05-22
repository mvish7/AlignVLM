from core.utils.process_sample import prep_sample_for_generation

def generate_text_from_sample(model, processor, sample, processed_sample=False, max_new_tokens=512, device="cuda"):
    """
    infer the model to generate text from image-text input
    """
    if not processed_sample:
        sample = prep_sample_for_generation(processor, sample, device, model.dtype)

    # Generate text with the model
    generated_ids = model.generate(**sample.to(model.dtype), max_new_tokens=max_new_tokens, do_sample=True)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(sample.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text # Return the first decoded output text