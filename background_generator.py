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


# Function to generate prompts using xAI API
def generate_prompt_by_description(description, element, num_count):
    """Use xAI API to generate creative background prompts."""
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1"
    )
    prompt = f"""This is the description of the user's image: <description>{description}</description>
These are the elements I provided: <element>{element}</element>
Combine the user's image description and the elements I provided (if there are none, you can create your own) to randomly generate {num_count} prompts.

The fixed format for prompts is  ```Fill the background environment for the user's uploaded image. Preserve the objects in the user's image as much as possible without modifications.Ensure the background filled in completely. Seamlessly blending the background and lighting of the user's image with its object. Ensure the background complements the style and mood of the original image naturally. The generated image is a high-resolution photograph. The generated image <content>```  Replace the <content> tag with you generate. The output should not include tags. Output as a JSON array with string only, with each element should be raw string."""
    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative assistant tasked with generating background prompts for images in a specific format. Do not include Markdown code fences (```json or ```), tags, or any additional text outside the JSON array. The response must be valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        response_text = response.choices[0].message.content.strip()
        logging.info(f"Generated prompts response: {response_text}")
        prompts = json.loads(response_text)
        if not isinstance(prompts, list) or len(prompts) != num_count:
            raise ValueError(
                f"Expected {num_count} prompts, but got {len(prompts)} or invalid format."
            )
        logging.info(f"Generated {num_count} prompts successfully:\n {prompts}.")
        return prompts
    except Exception as e:
        logging.error(f"Error calling xAI API: {e}")
        return []


# Function to describe image using xAI vision
def generate_prompt_by_image(image_file):
    base64_image = encode_image_to_base64(image_file)
    file_extension = os.path.splitext(image_file)[1].lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(file_extension, "image/jpeg")
    image_data_uri = f"data:{mime_type};base64,{base64_image}"

    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri, "detail": "high"},
                },
                {
                    "type": "text",
                    "text": "Provide a brief description of the user's image.",
                },
            ],
        },
    ]
    completion = client.chat.completions.create(
        model="grok-2-vision-latest",
        messages=messages,
        temperature=0.01,
    )
    return completion.choices[0].message.content


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


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


def generate_image_by_chat(prompt: str, image_file: str, num_count: int = 3):
    client, generate_content_config = _init_client()
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"
    first_prompt = "Provide a brief description of the user's image."

    file = client.files.upload(file=image_file)
    chat = client.chats.create(
        model="gemini-2.0-flash-exp", config=generate_content_config
    )
    response = chat.send_message(
        [
            types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type),
            types.Part.from_text(text=first_prompt),
        ]
    )

    if response.text:
        logging.info(f"Image {base_name} description: {response.text}")
        generated_prompts = generate_prompt_by_description(
            response.text, prompt, num_count
        )
        for p in generated_prompts:
            response = chat.send_message(p)
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    file_path = save_binary_file(
                        output_file_name,
                        part.inline_data.mime_type,
                        part.inline_data.data,
                    )
                    yield file_path
                elif part.text:
                    logging.info(f"Text output for {base_name}: {part.text}")


def generate_image(elements: str, image_file: str, num_count: int = 3):
    client, generate_content_config = _init_client()
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"
    description = generate_prompt_by_image(image_file)
    prompts = generate_prompt_by_description(description, elements, num_count)

    logging.info(f"Image {base_name} description: {description}")
    for prompt in prompts:
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
                logging.info(f"Text output for {base_name}: {part.text}")


def generate_image_simple(elements: str, image_file: str, num_count: int = 3):
    client, generate_content_config = _init_client()
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"

    logging.info(f"Image {base_name}")

    prompt = f"Fill the background environment for the user's uploaded image. Preserve the objects in the user's image as much as possible without modifications. Ensure the background filled in completely but respect the user's image's size and oritentaion and view angle, seamlessly blending the new background with the edges of the user's image. Ensure the background complements the style and mood of the original image naturally. The generated image is a high-resolution photograph.  pick some element of this list to generated image if you can: {elements}"

    prompt = f"Fill the background environment for the user's uploaded image."
    first_prompt = "Provide a brief description of the user's image."

    file = client.files.upload(file=image_file)
    chat = client.chats.create(
        model="gemini-2.0-flash-exp", config=generate_content_config
    )
    response = chat.send_message(
        [
            types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type),
            types.Part.from_text(text=prompt),
        ]
    )

    if num_count > 1:
        prompts = [] + [
            f"try another difference background,  pick some element of this list to generated image if you can: {elements}"
        ] * (num_count - 1)
    else:
        prompts = []

    if response.text:
        logging.info(f"Image {base_name} description: {response.text}")
    else:
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                file_path = save_binary_file(
                    output_file_name,
                    part.inline_data.mime_type,
                    part.inline_data.data,
                )
                yield file_path
    for p in prompts:
        response = chat.send_message(p)
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                file_path = save_binary_file(
                    output_file_name,
                    part.inline_data.mime_type,
                    part.inline_data.data,
                )
                yield file_path
            elif part.text:
                logging.info(f"Text output for {base_name}: {part.text}")


# Gradio UI function with streaming updates
def gradio_interface(image_files, prompt, choices, num_count):
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("XAI_API_KEY"):
        yield []  # Match output_gallery
        return

    if not image_files:
        yield []  # Match output_gallery
        return

    logging.info(f"Received image files: {image_files}")
    all_output_images = []

    # Process each image and yield results incrementally
    for image_file in image_files:
        if not os.path.isfile(image_file):
            logging.error(f"Invalid file path: {image_file}")
            continue
        logging.info(f"Processing image: {image_file}")
        if choices == "Simple":
            for image_path in generate_image_simple(
                prompt, image_file, num_count=num_count
            ):
                all_output_images.append(image_path)
                yield all_output_images  # Yield list for output_gallery
        elif choices == "Context":
            for image_path in generate_image_by_chat(
                prompt, image_file, num_count=num_count
            ):
                all_output_images.append(image_path)
                yield all_output_images  # Yield list for output_gallery
        else:
            for image_path in generate_image(
                elements=prompt, image_file=image_file, num_count=num_count
            ):
                all_output_images.append(image_path)
                yield all_output_images  # Yield list for output_gallery


# Function to update preview
def update_preview(image_files):
    return image_files if image_files else []


# Create Gradio UI
with gr.Blocks(title="Gemini Image Background Generator") as demo:
    gr.Markdown("# Gemini Image Background Generator")
    gr.Markdown(
        "Upload one or more images and provide a prompt to generate new backgrounds using xAI and Gemini APIs."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.File(
                label="Upload Images", file_count="multiple", type="filepath"
            )
            preview_gallery = gr.Gallery(
                label="Uploaded Images Preview", show_label=True
            )
            prompt_input = gr.Textbox(
                label="Prompt (Elements)",
                placeholder="e.g., lemons and a bottle of perfume",
                value="鱼、柠檬、苹果、橡木台面、木质橱柜，白色台面、黑色大理石台面",
            )
            num_count_input = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Number of Backgrounds per Image",
                info="Select how many background variations to generate for each image.",
            )
            choices = gr.Radio(
                label="Mode", choices=["Simple", "Normal", "Context"], value="Simple"
            )
            submit_button = gr.Button("Generate")

        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Images", show_label=True)

    # Update preview when images are uploaded
    image_input.change(fn=update_preview, inputs=image_input, outputs=preview_gallery)

    # Generate images with streaming updates
    submit_button.click(
        fn=gradio_interface,
        inputs=[image_input, prompt_input, choices, num_count_input],
        outputs=[output_gallery],
    )

if __name__ == "__main__":
     logging.info("Starting the Gradio UI.")
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("XAI_API_KEY"):
        print("Please set the GEMINI_API_KEY and XAI_API_KEY environment variables.")
    else:
        demo.launch()
    logging.info("Gradio UI execution completed.")
