# run it:  python main.py
# then type your messages. /exit (or ctrl+c) leaves

from config import *
from classes.agent import Agent
from functions.commands import check_input

if __name__ == "__main__":                           # runs only when you start main.py directly
    agent = Agent(MODEL, API_KEY, stream=STREAM, effort=EFFORT,       # by name: can never land
                  print_thinking=PRINT_THINKING)                  # in the wrong parameter
    print("model: " + MODEL + "   (/exit to leave)")

    try:
        while True:                                  # one round = one message from you, one answer from the model
            text = input("\n> ").strip()             # strip: no spaces or empty lines around it
            what = check_input(text)                 # "skip", "exit" or "message" (see functions/commands.py)
            if what == "exit":
                break                                # leave the loop: the program ends below
            if what == "skip":
                continue                             # nothing for the model: ask for the next input
            try:
                agent.ask(text)                      # ask prints the answer itself. the conversation remembers
                                                     # every round, so the model knows what was said before
            except Exception as e:                   # the model or the connection failed: say what happened,
                                                     # then go on chatting. \n first: a stream may have
                                                     # stopped in the middle of a line
                print("\nerror: " + type(e).__name__ + ": " + str(e))
    except (KeyboardInterrupt, EOFError):            # ctrl+c, or ctrl+d / the input ended: leave quietly
        print()

    print("bye")
