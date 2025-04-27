import random

from PIL import Image
system_message = """You are a Vision Language Model specialized in image captioning task. Please be thorough and 
descriptive in your captions. Your task is to analyze the provided image and respond with a detailed caption that
describes the content of the image."""

TEXTS = [
    "please describe what you observe in this image.",
    "give a detailed explanation of the scene in this picture.",
    "could you summarize the contents of this image in a few sentences?",
    "how would you describe this image to someone who can't see it?",
    "provide a detailed summary of what this picture shows.",
    "in a few sentences, explain what's happening in this image.",
    "could you briefly explain the scene captured in this image?",
    "describe the key elements and details visible in this picture.",
    "offer a short description of what you notice in this image.",
    "in 4-5 sentences, summarize the main aspects of this photo.",
    "give a quick overview of what this image is about.",
    "could you detail the important parts you observe in this picture?",
    "please share a concise summary of this image.",
    "write a few lines describing what you see in this photograph.",
    "summarize the overall scene depicted in this image.",
    "in 4-5 sentences, describe the important features of this image.",
    "briefly describe what is happening in this picture.",
    "what do you see in this image? describe it in a few sentences.",
    "explain what this image portrays in a short paragraph.",
    "give a clear and detailed description of this photo."
]


def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image_path"],
                },
                {
                    "type": "text",
                    "text": random.choice(TEXTS)
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["caption"]}],
        },
    ]


def gather_images_from_message(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and element.get("type") == "image":
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                    image = Image.open(image)
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs