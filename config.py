import os
from functions.read_json import read_json
from functions.environment import read_key

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))   # the folder this file lives in. .env is found from here
ENV_FILE = os.path.join(HERE, ".env")
MODEL_FILE = os.path.join(HERE, "settings", "model.json")
SOUL_FILE = os.path.join(HERE, "Ego", "SOUL.md")     # who the agent is: the start of its system prompt. you write it
TOOLS_DIR = os.path.join(HERE, "tools")              # tools/<name>.py = one tool each (see functions/load_tools.py)

# Model settings: read the file once, then take what we need from it
MODEL_SETTINGS = read_json(MODEL_FILE)
MODEL = MODEL_SETTINGS["Model"]                        # required: without a model nothing works
STREAM = MODEL_SETTINGS.get("Stream", False)           # optional: True = print the answer piece by piece as it arrives
EFFORT = MODEL_SETTINGS.get("Effort")                  # optional: how hard the model thinks ("low", "medium", "high").
                                                       # None = the model's own default
PRINT_THINKING = MODEL_SETTINGS.get("PrintThinking", False)   # optional: True = show the thinking too, not only the answer

# Provider name
provider = MODEL.split("/")[0].upper()               # "deepseek/deepseek-flash" -> "DEEPSEEK"

# API key for using the model from the provider
API_KEY = read_key(provider, ENV_FILE)
