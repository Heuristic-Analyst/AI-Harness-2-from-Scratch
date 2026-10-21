# commands.py — what you type is either a message for the model, or a command for the program.
# a command starts with "/". this file decides which one it is, so main.py only has to act on the answer.

COMMANDS = {                                         # every command, with what it does. the help list is made from
    "/exit": "leave the program",                    # this, so a new command only needs one new line here
    "/quit": "leave the program (the same as /exit)",
}


def help_text():
    """the list of all commands, one per line"""
    lines = ["commands:"]
    for command, what in COMMANDS.items():
        lines.append("  " + command.ljust(8) + what)    # ljust(8): fill up to 8 characters, so the texts line up
    return "\n".join(lines)


def check_input(text):
    """look at what the user typed (already stripped) and say what to do with it:
    "skip"     nothing to do: an empty line, or a command that was already handled (e.g. an unknown one)
    "exit"     leave the program
    "message"  not a command: send it to the model"""
    if text == "":                                   # just pressed enter
        return "skip"
    if not text.startswith("/"):                     # no "/" in front: a normal message
        return "message"

    command = text.split()[0]                        # the first word: "/exit now" -> "/exit"
    if command in ("/exit", "/quit"):
        return "exit"

    print("command not found: " + command)           # starts with "/", but we don't know it
    print(help_text())
    return "skip"
