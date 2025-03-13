import base64
import os
from google import genai
from google.genai import types
from google.genai import chats
from openai import OpenAI
import json
import logging
import argparse
import datetime
from typing import Tuple  # Added import for Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to save binary data (e.g., generated image)
def save_binary_file(base_name, mime_type, data):
    """Save binary data to a file with the specified name and MIME type."""
    extension = mime_type.split("/")[-1]
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_file_name = f"{base_name}_{current_datetime}.{extension}"
    with open(full_output_file_name, "wb") as f:
        f.write(data)
        logging.info(f"Saved file of MIME type {mime_type} to: {full_output_file_name}")


# Function to generate prompts using xAI API
def generate_prompt_by_description(description, element, num_count):
    """Use xAI API to generate creative background prompts."""
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1"
    )
    model = "grok-2-latest"
    # client = OpenAI(
    #     api_key=os.environ.get("OPENAI_API_KEY")
    # )
    # model = "gpt-4o-mini"
    prompt = f"""This is the description of the user's image: <description>{description}</description>
These are the elements I provided: <element>{element}</element>
Combine the user's image description and the elements I provided (if there are none, you can create your own) to randomly generate {num_count} prompts.

The fixed format for prompts is  ```Fill the background environment for the user's uploaded image. Preserve the objects in the user's image as much as possible without modifications.Ensure the background filled in completely. Seamlessly blending the background and lighting of the user's image with its object. Ensure the background complements the style and mood of the original image naturally. The generated image is a high-resolution photograph. The generated image <content>```  Replace the <content> tag with you generate. The output should not include tags. Output as a JSON array with string only, with each element should be raw string."""
    # Call the xAI API
    try:
        response = client.chat.completions.create(
            model="grok-beta",  # Adjust model name if updated
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative assistant tasked with generating background prompts for images in a specific format.  Do not include Markdown code fences (```json or ```), tags, or any additional text outside the JSON array. The response must be valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            # max_tokens=500,  # Increased to accommodate multiple prompts
            temperature=0.9,  # High creativity for varied backgrounds
        )
        # Extract the response content
        response_text = response.choices[0].message.content.strip()

        # Parse the response as JSON
        prompts = json.loads(response_text)

        # Validate that we got the expected number of prompts
        if not isinstance(prompts, list) or len(prompts) != num_count:
            raise ValueError(
                f"Expected {num_count} prompts, but got {len(prompts)} or invalid format."
            )

        logging.info(f"Generated {num_count} prompts successfully:\n {prompts}.")
        return prompts

    except Exception as e:
        logging.error(f"Error calling xAI API: {e}", e)
        return []


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def generate_prompt_by_image(image_file):
    # Check if the file exists
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found at: {image_file}")

    # Get the base64-encoded image
    base64_image = encode_image_to_base64(image_file)

    # Determine the MIME type based on file extension (adjust as needed)
    file_extension = os.path.splitext(image_file)[1].lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(file_extension, "image/jpeg")  # Default to JPEG if unknown

    # Create a data URI from the base64 string
    image_data_uri = f"data:{mime_type};base64,{base64_image}"

    # Initialize the xAI client
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1"
    )
    model = "grok-2-vision-latest"

    # Construct the message with the local image as a data URI
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri,
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "Provide a brief description of the user's image.",
                },
            ],
        },
    ]

    # Call the xAI API
    completion = client.chat.completions.create(
        model="grok-2-vision-latest",
        messages=messages,
        temperature=0.01,
    )

    # Print the response
    print(completion.choices[0].message.content)


def _init_client() -> Tuple[genai.Client, types.GenerateContentConfig]:
    # Initialize the Gemini API client with your API key
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    logging.info("Initialized Gemini API client.")

    # Configuration for content generation
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=1,
        automatic_function_calling={"disable": True},
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="OFF",  # Adjust safety settings as needed
            ),
        ],
        response_mime_type="text/plain",
    )
    logging.info("Configured content generation settings.")

    return client, generate_content_config


def generate_image_by_chat(prompt, image_file, num_count=3):
    client, generate_content_config = _init_client()

    base_name = os.path.splitext(os.path.basename(image_file))[0]

    output_file_name = f"{base_name}_modified"

    first_prompt = "Provide a brief description of the user's image."

    logging.info("image_file: %s", image_file)
    file = client.files.upload(file=image_file)

    chat: chats.Chats = client.chats.create(
        model="gemini-2.0-flash-exp",  # Replace with the correct model name
        config=generate_content_config,
    )
    response = chat.send_message(
        [
            types.Part.from_uri(
                file_uri=file.uri,
                mime_type=file.mime_type,
            ),
            types.Part.from_text(text=first_prompt),
        ]
    )

    if response.text:
        logging.info(f"Received text output:\n{response.text}")
        generated_prompts = generate_prompt_by_description(
            response.text, prompt, num_count
        )
        for prompt in generated_prompts:
            response = chat.send_message(prompt)
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    mime_type = part.inline_data.mime_type
                    save_binary_file(
                        output_file_name,
                        mime_type,
                        part.inline_data.data,
                    )
                elif part.text:
                    logging.info(f"Received text output: {part.text}")
                else:
                    logging.warning("Received an unexpected response part.")

    else:
        logging.warning("Received an unexpected response part.")


def generate_image(elements, image_file, num_count=3):
    client, generate_content_config = _init_client()

    # Derive the output file name from the input image file
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"

    description = generate_prompt_by_image(image_file)
    prompts = generate_prompt_by_description(description, elements, num_count)

    for prompt in prompts:
        # Upload the image file
        file = client.files.upload(file=image_file)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=file.uri,
                        mime_type=file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        logging.info("Defined content for API request.")

        # Generate the content and process the response
        logging.info(f"Starting content generation with file name: {output_file_name}")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Replace with the correct model name
            contents=contents,
            config=generate_content_config,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                mime_type = part.inline_data.mime_type
                save_binary_file(
                    output_file_name,
                    mime_type,
                    part.inline_data.data,
                )
            elif part.text:
                logging.info(f"Received text output: {part.text}")
            else:
                logging.warning("Received an unexpected response part.")


def setup_parser():
    parser = argparse.ArgumentParser(description="Generate an image using Gemini API.")
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt for image generation."
    )
    parser.add_argument(
        "--image_file",
        type=str,
        required=True,
        help="The path to the input image file.",
    )
    parser.add_argument(
        "--use_chat",
        action="store_true",
        help="Use chat-based image generation instead of direct generation.",
    )
    return parser


if __name__ == "__main__":
    logging.info("Starting the main execution.")
    # Ensure the API key is set in your environment
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        parser = setup_parser()
        args = parser.parse_args()
        if args.use_chat:
            generate_image_by_chat(args.prompt, args.image_file)
        else:
            generate_image(args.prompt, args.image_file)
    logging.info("Image generation completed.")
