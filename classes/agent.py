import json
import time
import threading
import litellm

litellm.drop_params = True     # a model that can't do something we ask for (e.g. thinking): leave it out, don't crash

RETRIES = 3                    # tries after the first one, only for errors that may go away by themselves
RETRY_ON = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
            litellm.InternalServerError, litellm.ServiceUnavailableError)

class Agent:
    def __init__(self, model, api_key, instructions=None, stream=False, effort=None, print_thinking=False,
                 tools=None, conversation=None):
        self.model = model
        self.api_key = api_key
        self.instructions = instructions       # who the agent is (the soul): the first part of its system prompt
        self.tools = tools or {}               # the tools it may use: name -> {"schema", "run"} (functions/load_tools.py)
        self.stream = stream                   # True = the answer is printed piece by piece while it arrives
        self.effort = effort                   # how hard the model thinks: "low", "medium", "high". None = its default
        self.print_thinking = print_thinking   # True = the thinking is printed too. False = only the answer
        self.printing = None                   # what kind of text was printed last: "THINKING", "ANSWER" or None
        self.tokens = 0                        # how big the conversation is, in tokens, as of the last answer.
                                               # 0 = no answer yet. the next call sends this + the new message
        self.conversation = conversation or []
        self.refresh_system()  # always. no instructions = no system message (see refresh_system)

    def build_system(self):
        system = self.instructions or ""  # 1. who the agent is. none given: start with an empty text
        return system

    def refresh_system(self):
        """build the system prompt, and put it in as the first message. an empty prompt means no system
        message at all: an old one is removed"""
        text = self.build_system()
        has_one = bool(self.conversation) and self.conversation[0]["role"] == "system"
        if not text:  # nothing to say: no system message at all
            if has_one:
                self.conversation.pop(0)  # an old one is removed, so it can't stay outdated
            return
        system = {"role": "system", "content": text}
        if has_one:
            self.conversation[0] = system
        else:
            self.conversation.insert(0, system)

    def ask(self, text):
        """send a message, and loop until the model answers without asking for a tool:
        model -> it asks for tools -> run them -> their results go back to the model -> model -> ...
        prints everything while it happens, and returns the final answer as text"""
        question = {"role": "user", "content": text}
        self.conversation.append(question)
        try:
            while True:                            # one round = one model call (+ the tools it asked for)
                response = self.call_model()
                response = self.show(response)     # inside the try: when streaming, the connection can also fail
                                                   # here, in the middle of the answer. afterwards: the finished response
                message = response.choices[0].message   # the answer: text, thinking, tool calls
                self.conversation.append(message.model_dump(exclude_none=True))   # the WHOLE message as a dict
                self.tokens = self.count_tokens(response)   # the conversation grew: how big is it now?
                if not message.tool_calls:         # CHECK: does the model want a tool?
                    return message.content or ""   #   no: that was the final answer. done
                self.run_tools(message.tool_calls) #   yes: run them, put the results in, and loop: ask the model again
        except Exception:
            if self.conversation[-1] is question:  # it failed before anything happened: take the question back,
                self.conversation.pop()            # so the list is as before. (if tools already ran, their work stays)
            raise                                  # and let the caller know. main.py prints it

    # --- tools: every tool call of one answer runs in its own thread, all at the same time ---
    def run_tools(self, tool_calls):
        """run all tool calls of one answer at the same time, wait until every one is done, then put the results
        into the conversation, in the order the model asked for them. each result carries the id of its call:
        that id is the only way the model knows which result belongs to which call"""
        results = {}                               # call id -> its result. every thread writes into this one dict
        threads = []
        for call in tool_calls:                    # 1. start one thread per call. they all run at once
            print("\nTOOL CALL: " + call.function.name + "(" + (call.function.arguments or "") + ")")
            thread = threading.Thread(target=self.run_tool, args=(call, results), daemon=True)
            thread.start()                         #    daemon: a running tool can't keep the program from ending
            threads.append(thread)
        for thread in threads:                     # 2. wait until EVERY one is done. they finish in any order,
            thread.join()                          #    that's why they write into a dict by id, not into a list
        for call in tool_calls:                    # 3. in the order the model ASKED, not the order they finished
            result = results.get(call.id, "error: the tool gave no result")   # never missing: see run_tool
            print("TOOL RESULT: " + call.function.name + " -> " + result)
            self.conversation.append({"role": "tool",
                                      "tool_call_id": call.id,   # which call this answers: must match exactly
                                      "name": call.function.name,
                                      "content": result})

    def run_tool(self, call, results):
        """run ONE tool call (in its own thread) and put its result into results[call.id]. it never fails:
        every problem becomes an error text for the model, so every call always gets an answer. (a call without
        an answer breaks the whole conversation: the provider rejects every request after it)"""
        name = call.function.name
        try:
            if name not in self.tools:             # the model asked for a tool it doesn't have
                raise Exception("there is no tool called '" + str(name) + "'. yours: " + (", ".join(self.tools) or "none"))
            args = json.loads(call.function.arguments or "{}")   # the arguments arrive as json TEXT: make a dict
            result = str(self.tools[name]["run"](self, **args))  # run(agent, expression=...). always text
        except Exception as e:                     # wrong arguments, broken json, the tool crashed: tell the model,
            result = "error: " + type(e).__name__ + ": " + str(e)   # so it can try again
        results[call.id] = result

    def count_tokens(self, response):
        """how big the conversation is now, in tokens: everything that was sent + what the model wrote back,
        as the provider counted it (response.usage). exact, and free: it comes with every answer.
        no count from the provider (some local models): litellm estimates it from the conversation instead"""
        usage = getattr(response, "usage", None)            # None if the response has no usage at all
        if usage and usage.total_tokens:                     # total = prompt_tokens (sent) + completion_tokens (written)
            return usage.total_tokens
        return litellm.token_counter(model=self.model, messages=self.conversation)   # an estimate, good enough

    def call_model(self):
        """one model call. errors that may go away by themselves (busy, network) are tried again, waiting
        longer each time. errors that won't (wrong key, bad request) are raised right away.
        streaming: this returns as soon as the answer STARTS to arrive. a failure after that is not retried
        here (half of the answer is printed already): it shows up in show(), and ask() handles it"""
        wait = 1                               # seconds before the first retry: 1, 2, 4
        for attempt in range(RETRIES + 1):
            try:
                return litellm.completion(model=self.model,
                                          api_key=self.api_key,
                                          messages=self.conversation,
                                          tools=[t["schema"] for t in self.tools.values()] or None,
                                                   # what the model may ask for. no tools: None
                                          stream=self.stream,
                                          stream_options={"include_usage": True} if self.stream else None,
                                                   # streaming: also send the token count, in one last chunk.
                                                   # (without streaming it always comes, and this must be None)
                                          reasoning_effort=self.effort,
                                          cache_control_injection_points=[{"location": "message", "index": -1}],
                                          timeout=120)
            except RETRY_ON as e:
                if attempt == RETRIES:         # the last try failed too: give up
                    raise
                print("[Error calling the model - " + type(e).__name__ + ", trying again in " + str(wait) + " s]")
                time.sleep(wait)
                wait *= 2

    def show(self, response):
        """print the model's thinking and answer, and return the finished response (the answer is in
        .choices[0].message, the token count in .usage).
        not streaming: the whole answer arrives at once, in one response, and is printed at once.
        streaming: the response is a stream of small pieces (chunks). each is printed the moment it arrives,
        then all of them are glued back together into one finished response, the same as without streaming"""
        self.printing = None                               # a new answer: nothing printed for it yet

        if not self.stream:
            message = response.choices[0].message          # message = the whole answer
            self.print_part("THINKING", getattr(message, "reasoning_content", None))   # None if it didn't think
            self.print_part("ANSWER", message.content)
            print()                                        # end the last line
            return response

        chunks = []                                        # every chunk, to glue the full message together at the end
        for chunk in response:                             # waits for each next chunk while the model writes
            chunks.append(chunk)
            if not chunk.choices:                          # some chunks are only bookkeeping, e.g. the last one
                continue                                   # with the token count. kept in chunks, nothing to print
            delta = chunk.choices[0].delta                 # delta = only what is new in this chunk
            self.print_part("THINKING", getattr(delta, "reasoning_content", None))   # a piece of thinking, or None
            self.print_part("ANSWER", delta.content)       # a piece of the answer, or None
        print()                                            # the answer is complete: end its line
        return litellm.stream_chunk_builder(chunks, messages=self.conversation)   # litellm glues the chunks into
                                                           # one response. messages: if no chunk had a token count,
                                                           # litellm counts the conversation itself

    def print_part(self, kind, text):
        """print one piece of THINKING or ANSWER text. a title appears whenever the kind changes"""
        if not text:                                       # no text of this kind in this piece: nothing to do
            return
        if kind == "THINKING" and not self.print_thinking: # thinking is hidden: skip it (it is still stored)
            return
        if kind != self.printing:                          # a different kind than the last piece: title first
            if self.printing:                              # something was printed before: an empty line between
                print("\n")
            print(kind + ":")
            self.printing = kind                           # remember: now we are in this kind
        print(text, end="", flush=True)                    # end="": no new line after each piece.
                                                           # flush: show it right now, not when python feels like it
