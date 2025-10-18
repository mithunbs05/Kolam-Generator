import requests
from dotenv import load_dotenv
import os

def remove_bg(input_path, output_path, api_key):
    """
    Removes the background from an image using Remove.bg API.
    
    :param input_path: Path to input image
    :param output_path: Path to save output image
    :param api_key: Your remove.bg API key
    """
    with open(input_path, "rb") as image_file:
        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": image_file},
            data={"size": "auto"},
            headers={"X-Api-Key": api_key},
        )
    
    if response.status_code == requests.codes.ok:
        with open(output_path, "wb") as out:
            out.write(response.content)
        print(f"✅ Background removed! Saved at: {output_path}")
    else:
        print(f"❌ Error: {response.status_code}, {response.text}")

# Example usage
api_key = os.getenv("black_bg_api_key") 
remove_bg("./real_images/real_image2.png", "./black_images/black.png", api_key)
