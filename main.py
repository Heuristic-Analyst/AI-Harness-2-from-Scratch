# run it:  python main.py

from config import *
from classes.agent import Agent

if __name__ == "__main__":                           # runs only when you start main.py directly
    agent = Agent(MODEL, API_KEY, stream=STREAM, effort=EFFORT,       # by name: can never land
                  print_thinking=PRINT_THINKING)                  # in the wrong parameter
    try:
        agent.ask(input("> "))                     # ask prints the answer itself now
    except Exception as e:  # the model or the connection failed: say what happened.
                            # \n first: a stream may have stopped in the middle of a line
        print("\nerror: " + type(e).__name__ + ": " + str(e))
