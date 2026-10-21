# run it:  python main.py

from config import *
from classes.agent import Agent

if __name__ == "__main__":                           # runs only when you start main.py directly
    agent = Agent(MODEL, API_KEY)
    try:
        print(agent.ask(input("> ")))
    except Exception as e:  # the model or the connection failed: say what happened
        print("error: " + type(e).__name__ + ": " + str(e))
