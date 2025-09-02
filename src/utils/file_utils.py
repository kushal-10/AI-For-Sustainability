"""Loading and saving files"""

import json

def load_json(json_file):
    with open(json_file) as json_file:
        return json.load(json_file)

def save_json(json_file, json_data):
    with open(json_file, "w") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

def load_text(text_file):
    with open(text_file) as text_file:
        return text_file.read()



