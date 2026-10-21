import os
import json
def read_model(file_path):
    """the line PROVIDER_API_KEY=... from .env, for the provider in MODEL. None if there is none
    (fine for local models like ollama). the key never goes into the environment"""
    if not os.path.isfile(file_path):                     # no .env at all: no key, no crash
        return None
    with open(file_path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as e:
            raise ValueError(file_path + " is not valid json: " + str(e))