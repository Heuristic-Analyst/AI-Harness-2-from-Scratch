# tools/results/search_result.py — like grep, on one saved result (functions/results.py): every line that contains
# a text, with the part it is in and its line number. the parts are glued back together first, so a line that was
# cut between two parts is still found whole.
# a long answer from this tool is saved as a result itself. that is on purpose: search that one again, narrower.

from functions import results

AROUND = 150                                         # a very long line (e.g. one-line json) is shown only this many
                                                     # characters before and after the match


def run(agent, name, text):
    parts = results.load(agent.results_dir, name)    # only this conversation's results folder
    if parts is None:
        return results.not_found(agent.results_dir, name)
    if not text.strip():
        return "give a text to search for"

    whole = ""                                       # the parts glued back together
    ends = []                                        # where each part ends in it, to say which part a line is in.
    for number in range(1, len(parts) + 1):          # taken from the real parts, so it stays right even if
        whole += parts[str(number)]                  # RESULT_PART was changed since this result was saved
        ends.append(len(whole))

    wanted = text.lower()                            # upper or lower case doesn't matter
    hits = []
    position = 0                                     # where the current line starts in the glued text
    for number, line in enumerate(whole.split("\n"), 1):   # enumerate(..., 1): line numbers, counted from 1
        at = line.lower().find(wanted)               # where in the line it is, or -1 if it isn't
        if at >= 0:
            part = 1
            while position + at >= ends[part - 1]:   # which part the match is in
                part += 1
            shown = line
            if len(line) > 2 * AROUND + len(text):   # a very long line: only the piece around the match
                shown = "..." + line[max(0, at - AROUND):at + len(text) + AROUND] + "..."
            hits.append("part " + str(part) + ", line " + str(number) + ": " + shown)
        position += len(line) + 1                    # + 1: the "\n" that split() took away
    if not hits:
        return "no line in " + name + " contains '" + text + "'"
    return (str(len(hits)) + " line(s) in " + name + " contain '" + text + "' (upper or lower case doesn't matter). "
            "read_result shows a whole part:\n" + "\n".join(hits))


TOOL = {
    "name": "search_result",
    "description": "Find lines in a saved result, like grep: every line that contains a text, with its part and "
                   "line number. Use it to find what you need in a big result, then read_result that part. "
                   "If the answer is long itself, it is saved as a new result: search that one again, narrower.",
    "parameters": {"type": "object",
                   "properties": {"name": {"type": "string", "description": "the result's name, e.g. results-3"},
                                  "text": {"type": "string", "description": "what to look for. upper or lower case "
                                                                            "doesn't matter"}},
                   "required": ["name", "text"]},
}
