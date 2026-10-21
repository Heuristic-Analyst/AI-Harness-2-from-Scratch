import time
import litellm

litellm.drop_params = True     # a model that can't do something we ask for (e.g. thinking): leave it out, don't crash

RETRIES = 3                    # tries after the first one, only for errors that may go away by themselves
RETRY_ON = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
            litellm.InternalServerError, litellm.ServiceUnavailableError)

class Agent:
    def __init__(self, model, api_key, instructions=None, stream=False, effort=None, print_thinking=False, conversation=None):
        self.model = model
        self.api_key = api_key
        self.instructions = instructions       # who the agent is (the soul): the first part of its system prompt
        self.stream = stream                   # True = the answer is printed piece by piece while it arrives
        self.effort = effort                   # how hard the model thinks: "low", "medium", "high". None = its default
        self.print_thinking = print_thinking   # True = the thinking is printed too. False = only the answer
        self.printing = None                   # what kind of text was printed last: "THINKING", "ANSWER" or None
        self.conversation = conversation or []
        if instructions:
            self.refresh_system()                  # the system prompt goes in as the very first message

    # --- the system prompt: always built fresh from its parts, never edited in place ---
    def build_system(self):
        """put the system prompt together from its parts and return it as text. it only builds, it changes
        nothing: refresh_system puts it into the conversation. the same parts always give exactly the same
        text, so the provider's cache of the conversation stays valid"""
        system = self.instructions             # 1. who the agent is
                                               # more parts are added here later (memory, skills, mode, ...),
                                               # always in the same order
        return system

    def refresh_system(self):
        """build the system prompt, and put it in as the first message of the conversation: replace the old one
        if there is one, otherwise add it at the very front"""
        system = {"role": "system", "content": self.build_system()}
        if self.conversation and self.conversation[0]["role"] == "system":   # there is one already: replace it
            self.conversation[0] = system
        else:                                                                # none yet: put it in front
            self.conversation.insert(0, system)

    def ask(self, text):
        """send a message, print the answer while it arrives, and return it as text.
        the model only knows what is in `conversation`"""
        question = {"role": "user", "content": text}
        self.conversation.append(question)
        try:
            response = self.call_model()
            message = self.show(response)      # inside the try: when streaming, the connection can also fail
                                               # here, in the middle of the answer
        except Exception:
            self.conversation.pop()            # it failed: take the question back, so the list is as before
            raise                              # and let the caller know. main.py prints it
        self.conversation.append(message.model_dump(exclude_none=True))   # the WHOLE message as a dict: answer,
                                                                          # thinking, and later tool calls
        return message.content or ""           # None if the model wrote no text: then an empty answer

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
                                          stream=self.stream,
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
        """print the model's thinking and answer, and return the finished message.
        not streaming: the whole answer arrives at once, in one response, and is printed at once.
        streaming: the response is a stream of small pieces (chunks). each is printed the moment it arrives,
        then all of them are glued back together into one finished message, the same as without streaming"""
        self.printing = None                               # a new answer: nothing printed for it yet

        if not self.stream:
            message = response.choices[0].message          # message = the whole answer
            self.print_part("THINKING", getattr(message, "reasoning_content", None))   # None if it didn't think
            self.print_part("ANSWER", message.content)
            print()                                        # end the last line
            return message

        chunks = []                                        # every chunk, to glue the full message together at the end
        for chunk in response:                             # waits for each next chunk while the model writes
            chunks.append(chunk)
            if not chunk.choices:                          # some chunks are only bookkeeping (e.g. token usage)
                continue
            delta = chunk.choices[0].delta                 # delta = only what is new in this chunk
            self.print_part("THINKING", getattr(delta, "reasoning_content", None))   # a piece of thinking, or None
            self.print_part("ANSWER", delta.content)       # a piece of the answer, or None
        print()                                            # the answer is complete: end its line
        return litellm.stream_chunk_builder(chunks).choices[0].message   # litellm glues the chunks into one reply

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
