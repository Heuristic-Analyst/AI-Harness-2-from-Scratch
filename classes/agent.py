import time
import litellm

RETRIES = 3                    # tries after the first one, only for errors that may go away by themselves
RETRY_ON = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
            litellm.InternalServerError, litellm.ServiceUnavailableError)

class Agent:
    def __init__(self, model, api_key, stream=False, conversation=None):
        self.model = model
        self.api_key = api_key
        self.stream = stream                   # True = the answer is printed piece by piece while it arrives
        self.conversation = []
        if conversation:
            self.conversation = conversation

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
        answer = message.content or ""         # None if the model wrote no text: then an empty answer
        self.conversation.append({"role": "assistant", "content": answer})
        return answer

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
                                          timeout=120)
            except RETRY_ON as e:
                if attempt == RETRIES:         # the last try failed too: give up
                    raise
                print("[Error calling the model - " + type(e).__name__ + ", trying again in " + str(wait) + " s]")
                time.sleep(wait)
                wait *= 2

    def show(self, response):
        """print the model's answer, and return the finished message (its text is in .content).
        not streaming: the whole answer arrives at once, in one response, and is printed at once.
        streaming: the response is a stream of small pieces (chunks). each is printed the moment it arrives,
        then all of them are glued back together into one finished message, the same as without streaming"""
        if not self.stream:
            message = response.choices[0].message          # message = the whole answer
            print(message.content or "")
            return message

        chunks = []                                        # every chunk, to glue the full message together at the end
        for chunk in response:                             # waits for each next chunk while the model writes
            chunks.append(chunk)
            if not chunk.choices:                          # some chunks are only bookkeeping (e.g. token usage)
                continue
            piece = chunk.choices[0].delta.content         # delta = only what is new in this chunk
            if piece:                                      # None while the model thinks or calls a tool
                print(piece, end="", flush=True)           # end="": no new line after each piece.
                                                           # flush: show it right now, not when python feels like it
        print()                                            # the answer is complete: end its line
        return litellm.stream_chunk_builder(chunks).choices[0].message   # litellm glues the chunks into one reply
