import time
import openai
from IPython import embed

OPENAI_KEYS = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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'
]



# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self):
        return ""



class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, **kwargs):
        super().__init__(model_name, api_key)
        self.kwargs = kwargs
    
    @staticmethod
    def parse_response(response, parse_fn):
        to_return = []
        for _, g in enumerate(response['choices']):
            text = g['text']
            text = parse_fn(text)
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))

        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts

    def generate(self, prompt, parse_fn):
        get_results = False
        while not get_results:
            embed()
            input()
            try:
                result = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    api_key=self.api_key,
                    **self.kwargs
                )
                get_results = True
            except Exception as e:
                if "exceeded your current quota" in e._message:
                    raise e
                else:
                    print('{} That is Rate Limit, sleep 15s and retry...'.format(e._message))
                    time.sleep(15)
                    continue
                
        return self.parse_response(result, parse_fn)