import litellm

class Agent:
    def __init__(self, model, api_key, conversation=None):
        self.model = model
        self.api_key = api_key
        self.conversation = []
        if conversation:
            self.conversation = conversation

    def ask(self, text):
        self.conversation.append({"role": "user", "content": text})
        """send one message, get the answer back as text. the model only knows what is in `messages`"""
        response = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            messages=self.conversation,
        )
        answer = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": answer})
        return answer
