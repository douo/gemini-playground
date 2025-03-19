import base64
import os
from google import genai
from google.genai import types
from google.genai import chats
from openai import OpenAI
import json
import logging
import datetime
import gradio as gr
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to save binary data (e.g., generated image)
def save_binary_file(base_name, mime_type, data):
    """Save binary data to a file with the specified name and MIME type."""
    # get base path from env
    base_path = os.environ.get("OUTPUT_PATH", "./output")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    extension = mime_type.split("/")[-1]
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_file_name = f"{base_path}/{base_name}_{current_datetime}.{extension}"
    with open(full_output_file_name, "wb") as f:
        f.write(data)
    logging.info(f"Saved file of MIME type {mime_type} to: {full_output_file_name}")
    return full_output_file_name


def _init_client() -> Tuple[genai.Client, types.GenerateContentConfig]:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    logging.info("Initialized Gemini API client.")
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=1,
        automatic_function_calling={"disable": True},
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"
            ),
        ],
        response_mime_type="text/plain",
    )
    logging.info("Configured content generation settings.")
    return client, generate_content_config


def generate_image(image_file: str, prompt: str, num_session, num_count):
    client, generate_content_config = _init_client()
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"

    logging.info(f"Generating {num_count} images for {base_name}...")

    file = client.files.upload(file=image_file)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    for i in range(num_session):
        logging.info(f"Starting session {i + 1} of {num_session} for {base_name}...")
        if num_count == 1:
            yield from _generate_single_image(
                image_file,
                prompt,
                client,
                generate_content_config,
                contents,
                output_file_name,
                base_name,
            )
        else:
            for image_path in _generate_multi_image(
                image_file,
                prompt,
                num_count,
                client,
                generate_content_config,
                contents,
                output_file_name,
                base_name,
            ):
                yield image_path
        logging.info(f"Completed session {i + 1} of {num_session} for {base_name}.")
    logging.info(f"Completed generating {num_count} images for {base_name}.")


def _generate_single_image(
    image_file: str,
    prompt: str,
    client,
    generate_content_config,
    contents,
    output_file_name,
    base_name,
):
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=contents,
        config=generate_content_config,
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            file_path = save_binary_file(
                output_file_name, part.inline_data.mime_type, part.inline_data.data
            )
            yield file_path
        elif part.text:
            logging.warning(f"Text output for {base_name}: {part.text}")


def _generate_multi_image(
    image_file: str,
    prompt: str,
    num_count,
    client,
    generate_content_config,
    contents,
    output_file_name,
    base_name,
):
    chat = client.chats.create(
        model="gemini-2.0-flash-exp", config=generate_content_config
    )

    message = contents[0].parts
    image_generated = 0

    while image_generated < num_count:
        logging.info(
            f"Generating next image. {num_count - image_generated} images remaining for {base_name}."
        )
        response = chat.send_message(message)
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                file_path = save_binary_file(
                    output_file_name,
                    part.inline_data.mime_type,
                    part.inline_data.data,
                )
                image_generated += 1
                yield file_path
            elif part.text:
                logging.warning(f"Text output for {base_name}: {part.text}")
        message = f"Please draw {num_count - image_generated} more independent images with different variations. based on the previous prompt."
        logging.info("message: " + message)

    logging.info(
        f"Total {image_generated} images generated for {base_name} in this session."
    )


def gradio_interface(image_file, prompt, num_session, num_count):
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("XAI_API_KEY"):
        yield []  # Match output_gallery
        return

    if not image_file or not prompt:
        yield []
        return
    all_output_images = []

    for image_path in generate_image(image_file, prompt, num_session, num_count):
        all_output_images.append(image_path)
        yield all_output_images


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="filepath"),
        "text",
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of sessions"),
        gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=1,
            label="Number of images to generate in each session",
        ),
    ],
    outputs=[gr.Gallery(label="Generated Images", show_label=True)],
)

if __name__ == "__main__":
    logging.info("Starting the Gradio UI.")
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variables.")
    else:
        demo.launch()
    logging.info("Gradio UI execution completed.")
