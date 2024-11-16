from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torchvision
import re

import pandas as pd
import requests
from PIL import Image, ImageOps
from io import BytesIO
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map="auto"
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}



def convert_text(received_text):
    # Define the expanded entity unit map
    expanded_entity_unit_map = {
        'width': {
            'centimetre': ['cm', 'centimeter', 'centimetre', 'centi', 'CM'],
            'foot': ['ft', 'foot', 'FT', 'Feet', 'feet'],
            'inch': ['in', 'inch', 'inches', 'IN'],
            'metre': ['m', 'meter', 'metre', 'M'],
            'millimetre': ['mm', 'millimeter', 'millimetre', 'MM'],
            'yard': ['yd', 'yard', 'Yard', 'YD']
        },
        'depth': {
            'centimetre': ['cm', 'centimeter', 'centimetre', 'centi', 'CM'],
            'foot': ['ft', 'foot', 'FT', 'Feet', 'feet'],
            'inch': ['in', 'inch', 'inches', 'IN'],
            'metre': ['m', 'meter', 'metre', 'M'],
            'millimetre': ['mm', 'millimeter', 'millimetre', 'MM'],
            'yard': ['yd', 'yard', 'Yard', 'YD']
        },
        'height': {
            'centimetre': ['cm', 'centimeter', 'centimetre', 'centi', 'CM'],
            'foot': ['ft', 'foot', 'FT', 'Feet', 'feet'],
            'inch': ['in', 'inch', 'inches', 'IN'],
            'metre': ['m', 'meter', 'metre', 'M'],
            'millimetre': ['mm', 'millimeter', 'millimetre', 'MM'],
            'yard': ['yd', 'yard', 'Yard', 'YD']
        },
        'item_weight': {
            'gram': ['g', 'gm', 'gms', 'grams', 'gram', 'G', 'Gr', 'GRM', 'grms', 'grm'],
            'kilogram': ['kg', 'kgs', 'kilogram', 'kilograms', 'KG', 'kilo'],
            'microgram': ['mcg', 'μg', 'microgram', 'micrograms', 'MG'],
            'milligram': ['mg', 'milligram', 'milligrams', 'MG'],
            'ounce': ['oz', 'ounce', 'ounces', 'OZ'],
            'pound': ['lb', 'lbs', 'pound', 'pounds', 'LB'],
            'ton': ['t', 'ton', 'tons', 'T']
        },
        'maximum_weight_recommendation': {
            'gram': ['g', 'gm', 'gms', 'grams', 'gram', 'G', 'Gr', 'GRM', 'grms', 'grm'],
            'kilogram': ['kg', 'kgs', 'kilogram', 'kilograms', 'KG', 'kilo'],
            'microgram': ['mcg', 'μg', 'microgram', 'micrograms', 'MG'],
            'milligram': ['mg', 'milligram', 'milligrams', 'MG'],
            'ounce': ['oz', 'ounce', 'ounces', 'OZ'],
            'pound': ['lb', 'lbs', 'pound', 'pounds', 'LB'],
            'ton': ['t', 'ton', 'tons', 'T']
        },
        'voltage': {
            'kilovolt': ['kV', 'kilovolt', 'kilovolts', 'KV'],
            'millivolt': ['mV', 'millivolt', 'millivolts', 'MV'],
            'volt': ['V', 'volt', 'volts']
        },
        'wattage': {
            'kilowatt': ['kW', 'kilowatt', 'kilowatts', 'KW'],
            'watt': ['W', 'watt', 'watts', 'WATT']
        },
        'item_volume': {
            'centilitre': ['cl', 'centilitre', 'centiliters', 'CL'],
            'cubic foot': ['cu ft', 'cubic foot', 'cubic feet', 'CF'],
            'cubic inch': ['cu in', 'cubic inch', 'cubic inches', 'CI'],
            'cup': ['cup', 'cups', 'C'],
            'decilitre': ['dl', 'decilitre', 'deciliters', 'DL'],
            'fluid ounce': ['fl oz', 'fluid ounce', 'fluid ounces', 'FLOZ'],
            'gallon': ['gal', 'gallon', 'gallons', 'GAL'],
            'imperial gallon': ['imp gal', 'imperial gallon', 'imperial gallons'],
            'litre': ['l', 'litre', 'liter', 'litres', 'L'],
            'microlitre': ['μl', 'microlitre', 'microliters', 'MICROL'],
            'millilitre': ['ml', 'millilitre', 'milliliter', 'milliliters', 'ML'],
            'pint': ['pt', 'pint', 'pints', 'PT'],
            'quart': ['qt', 'quart', 'quarts', 'QT']
        }
    }
    # Remove curly braces and extra spaces
    received_text = received_text.strip('{} ').strip()

    # Extract value and unit using regex
    match = re.match(r'(\d+(\.\d+)?)\s*(\w+)', received_text)
    if not match:
        return ""

    value, unit = match.group(1), match.group(3)

    # Find the correct unit from the expanded_entity_unit_map
    for entity, units in expanded_entity_unit_map.items():
        for key, aliases in units.items():
            if unit in aliases:
                return f"{value} {key}"

    return ""


def run_inference(image, entity_name, possible_units):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"""Can you tell me the {entity_name} of this product given in the image. Just return the
                     answer strictly in the format - {{value unit}} for example {{10.0 kilogram}}. Don't add anything
                     else in the response. Here are all the {{'possible_units': {possible_units}}}, In the output the
                     spelling of the unit must exactly match with one of the units given in the possible_units map, if not then convert it
                     to match one of the units in possible_units , the
                      value of which you detected from the image. If you can't find it, just return an empty string""",
                },
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        return None


# image = "/content/images of project.jpg"
entity_name = "item_weight"
possible_units = entity_unit_map[entity_name]

# output = convert_text(run_inference(image, entity_name, possible_units))

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # if grayscale:
        #     image = ImageOps.grayscale(image)

        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
        return None



image_url = "https://m.media-amazon.com/images/I/91LPf6OjV9L.jpg"


image = download_image(image_url)

output = convert_text(run_inference(image, entity_name, possible_units))

print(output)