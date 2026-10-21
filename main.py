# main.py — step 1: send one message to a model and print its answer. nothing else yet.
# run it:  python main.py
# needs:   pip install -r requirements.txt, and a .env file next to this one with your key, e.g.
#          DEEPSEEK_API_KEY=sk-...

import os
import litellm
from functions.model import read_model
from functions.environment import read_key
from classes.agent import Agent

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))   # the folder this file lives in. .env is found from here
ENV_FILE = os.path.join(HERE, ".env")
MODEL_FILE = os.path.join(HERE, "config", "model.json")

# Model name
MODEL = read_model(MODEL_FILE)["Model"]

# Provider name
provider = MODEL.split("/")[0].upper()               # "deepseek/deepseek-flash" -> "DEEPSEEK"

# API key for using the model from the provider
API_KEY = read_key(provider, ENV_FILE)

if __name__ == "__main__":                           # runs only when you start main.py directly
    agent = Agent(MODEL, API_KEY)
    print(agent.ask(input("> ")))
