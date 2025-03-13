import base64
import os
from google import genai
from google.genai import types
import logging
import argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to save binary data (e.g., generated image)
def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)


def generate_image(prompt, image_file):
    # Derive the output file name from the input image file
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    output_file_name = f"{base_name}_modified"

    # Initialize the Gemini API client with your API key
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    logging.info("Initialized Gemini API client.")

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

    # Configuration for content generation
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
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

    # Generate the content and process the response
    logging.info(f"Starting content generation with file name: {output_file_name}")
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash-exp",  # Replace with the correct model name
        contents=contents,
        config=generate_content_config,
    ):
        if (
            not chunk.candidates
            or not chunk.candidates[0].content
            or not chunk.candidates[0].content.parts
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            mime_type = chunk.candidates[0].content.parts[0].inline_data.mime_type
            extension = mime_type.split("/")[-1]
            full_output_file_name = f"{output_file_name}.{extension}"
            save_binary_file(
                full_output_file_name, chunk.candidates[0].content.parts[0].inline_data.data
            )
            logging.info(f"Saved image file: {full_output_file_name}")
            print(
                f"File of mime type {mime_type} saved to: {full_output_file_name}"
            )
        else:
            logging.info(f"Received text output: {chunk.text}")
            print(chunk.text)


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
    return parser


if __name__ == "__main__":
    logging.info("Starting the main execution.")
    # Ensure the API key is set in your environment
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        parser = setup_parser()
        args = parser.parse_args()

        generate_image(args.prompt, args.image_file)
    logging.info("Image generation completed.")
