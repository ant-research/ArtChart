from typing import Union, Any, Optional, Dict
from PIL import Image
from io import BytesIO
import magic
import megfile
import base64
import os
import json
import re

def image2byte(image: Image.Image) -> bytes:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def encode_image(image: Union[Image.Image, bytes, str]) -> str:
    if isinstance(image, str):
        image = Image.open(megfile.smart_open(image, 'rb')).convert('RGB')
        image = image2byte(image)
    elif isinstance(image, Image.Image):
        image = image2byte(image)

    mime_type = magic.from_buffer(image, mime=True)
    base64_encoded_data = base64.b64encode(image).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def parse_llm_output(input_string: str) -> Optional[Dict]:
    delimiter = '||V^=^V||'
    if input_string.count(delimiter) == 2:
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            start_index = input_string.find('[')
            end_index = input_string.rfind(']') + 1
            if re.match(r'^\[\d+, ?\d+\]$', input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {'score': scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string):
                scores = [int(input_string)]
                json_content = {'score': scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print("Failed to find the json content in the string.")
                return None
    if start_index != -1 and end_index != -1 and start_index != end_index:
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        try:
            new_data = json.loads(json_str)
        except:
            try:
                new_data = json.loads(fix_json(json_str))
                return new_data
            except:
                print("Error: Cannot fix", json_str)
                return None
        return new_data
    else:
        print("The required delimiters were not found correctly in the string.")
        return None

def fix_json(input_str: str) -> str:
    fixed_str = re.sub(r'(\w+):', r'"\1":', input_str)
    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = f'[{value}]'
        elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
            pass
        else:
            value = f'"{value}"'
        return f'{key}: {value}{comma}'
    fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)
    return fixed_str

def is_int_between_0_and_10(s):
    try:
        num = int(s)
        return 0 <= num <= 10
    except ValueError:
        return False

def prepare_prompt(prompt_template: str, **kwargs) -> str:
    prompt = prompt_template
    for k, v in kwargs.items():
        prompt = prompt.replace(f'<{k}>', str(v))
    return prompt
