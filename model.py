! pip install torch
! pip install git+https://github.com/huggingface/transformers
! pip install git+https://github.com/huggingface/accelerate
! pip install qwen-vl-utils
!pip install Pillow
!pip install pandas


from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map="auto"
)


# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
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

import re

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



import pandas as pd
import requests
from PIL import Image, ImageOps
from io import BytesIO
import torch




# image = ImageOps.grayscale(image)

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


from PIL import Image
import numpy as np
import pandas as pd


def process_dataframe():
  csv_path = input("enter the path of the csv file: ")
  test_df = pd.read_csv(csv_path)
  print(test_df.head())
  output_df = test_df

  url = input("enter the column name of the url: ")
  entity_type = input("enter the column name of the entity type: ")
  index_1 = input("enter the column name of the index: ")
  
  output_folder_path = input("enter the output folder path: ")
  for index, row in test_df.iterrows():
      image_url = row[url]
      entity_name = row[entity_type]
      corresponding_index = row[index_1]

      # Check if entity_name is valid
      if entity_name not in entity_unit_map:
          print(f"Invalid entity name: {entity_name}")
          output_df.at[index, "output"] = np.nan
          continue
      possible_units = entity_unit_map[entity_name]
      # Download image
      # image = download_image(image_url, grayscale=False)
      image = Image.open(image_url)
      if image is None:
          output_df.at[index, "output"] = np.nan
          continue

      # Run inference with timeout
      output = convert_text(run_inference(image, entity_name, possible_units))
      print(f"Completed index {corresponding_index} - total {index} - output: {output}")
      output_df.at[index, "output"] = output

        # Save to CSV every 10th interval
        
  try:
      output_df.to_csv(output_folder_path, index=False)
      print("output file is exported")
  except Exception as e:
      print(f"Error saving to CSV: {e}")





# prompt: give the code to store the image url in the csv format which is inside the folder
# the csv format should contain the columns below
# slno,image_id,image_path,entity_type
# the method should be in the below format the value in the entity_type column should be filled with  the entity_type value stored in that variable
# def folder_to_csv(folder_path,entity_type):

import os
import csv

def folder_to_csv(folder_path, entity_type):
    """
    Stores image URLs from a folder into a CSV file.

    Args:
        folder_path: The path to the folder containing the images.
        entity_type: The value to be filled in the 'entity_type' column.
    """

    csv_file_path = 'image_data.csv'  # Name of the output CSV file

    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['slno', 'image_id', 'image_path', 'entity_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row

        slno = 1
        image_id =1
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                # image_id = filename  # Use filename as image_id
                image_path = os.path.join(folder_path, filename) # Store the full path

                writer.writerow({
                    'slno': slno,
                    'image_id': image_id,
                    'image_path': image_path,
                    'entity_type': entity_type
                })
                slno += 1
                image_id +=1

# Example usage:
folder_path = '/content/product_images'  # Replace with the actual folder path
entity_type_value = 'item_weight'  # Replace with the desired entity type

folder_to_csv(folder_path, entity_type_value)


# prompt: give the code to display the csv file

import pandas as pd

def display_csv(file_path):
  """Displays the contents of a CSV file as a pandas DataFrame.

  Args:
    file_path: The path to the CSV file.
  """
  try:
    df = pd.read_csv(file_path)
    print(df)
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
  except pd.errors.EmptyDataError:
    print(f"Error: The CSV file at {file_path} is empty.")
  except pd.errors.ParserError:
    print(f"Error: Could not parse the CSV file at {file_path}. Check its format.")
  except Exception as e:
    print(f"An error occurred: {e}")


# Example usage (assuming the CSV is named 'image_data.csv' in the current directory)
display_csv('final.csv')
























