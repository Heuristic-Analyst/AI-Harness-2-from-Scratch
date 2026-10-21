import time
import litellm

RETRIES = 3                    # tries after the first one, only for errors that may go away by themselves
RETRY_ON = (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
            litellm.InternalServerError, litellm.ServiceUnavailableError)

class Agent:
    def __init__(self, model, api_key, conversation=None):
        self.model = model
        self.api_key = api_key
        self.conversation = []
        if conversation:
            self.conversation = conversation

    def ask(self, text):
        """send a message, get the answer back as text. the model only knows what is in `conversation`"""
        question = {"role": "user", "content": text}
        self.conversation.append(question)
        try:
            response = self.call_model()
        except Exception:
            self.conversation.pop()            # it failed: take the question back, so the list is as before
            raise                              # and let the caller know. main.py prints it
        answer = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": answer})
        return answer

    def call_model(self):
        """one model call. errors that may go away by themselves (busy, network) are tried again, waiting
        longer each time. errors that won't (wrong key, bad request) are raised right away"""
        wait = 1                               # seconds before the first retry: 1, 2, 4
        for attempt in range(RETRIES + 1):
            try:
                return litellm.completion(model=self.model,
                                          api_key=self.api_key,
                                          messages=self.conversation,
                                          timeout=120)
            except RETRY_ON as e:
                if attempt == RETRIES:         # the last try failed too: give up
                    raise
                print("[Error calling the model - " + type(e).__name__ + ", trying again in " + str(wait) + " s]")
                time.sleep(wait)
                wait *= 2
