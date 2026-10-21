# run it:  python main.py

from config import *
from classes.agent import Agent

if __name__ == "__main__":                           # runs only when you start main.py directly
    agent = Agent(MODEL, API_KEY)
    print(agent.ask(input("> ")))
