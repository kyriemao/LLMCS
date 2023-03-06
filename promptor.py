from IPython import embed
import re


def build_promptor(prompt_type):
    if prompt_type == "Rewrite":
        return RewritePrompter()
    elif prompt_type == "Rewrite-And-Response":
        return RewriteAndResponsePromptor()
    elif prompt_type == "CoT-Rewrite":
        return CoTRewritePromptor()
    elif prompt_type == "CoT-Rewrite-And-Response":
        return CoTRewriteAndResponsePromptor()
    elif prompt_type == "Rewrite-Then-Response":
        return RewriteThenResponsePromptor()
    else:
        raise NotImplementedError
    
class RewritePrompter:
    def __init__(self) -> None:
        self.instruction = "Reformulate the current question into a de-contextualized rewrite under the multi-turn information-seeking dialog context."
        self.stop_tokens = ['\n']
    
    def build_demo_prompt(self, dialogs: list):
        demo_prompt = []
        demo_prompt.append(self.instruction)
        
        for dialog in dialogs:
            dialog_prompt = []
            turns = dialog['turns']
            for turn in turns:
                turn_prompt = self.build_turn_prompt(turn, is_demo=True)
                dialog_prompt.append(turn_prompt) 
            demo_prompt.append("\n".join(dialog_prompt))
            
        return "\n\n".join(demo_prompt)
                
                
    def build_turn_prompt(self, turn, is_demo):
        turn_prompt = []
        turn_prompt.append("Turn: {}".format(turn['turn_id']))
        turn_prompt.append("Question: {}".format(turn['question']))
        if is_demo:
            turn_prompt.append("Rewrite: {}".format(turn['manual_rewrite']))
            turn_prompt.append("Response: {}".format(turn['response']))
        else: # for test
            turn_prompt.append("Rewrite: ")
        
        return "\n".join(turn_prompt)
        
        
    def build_this_turn_prompt_for_prediction(self, pre_prompt, this_turn, last_predicted_rewrite, last_response):
        pre_prompt_components = pre_prompt.split("\n\n")

        # update the last turn of the last dialog's info in the prompt
        if last_predicted_rewrite is not None:
            last_dialog_prompt = pre_prompt_components[-1]
            pre_prompt_components.pop()
            last_dialog_prompt_turns = last_dialog_prompt.split('\n')
            last_dialog_prompt_turns[-1] = "Rewrite: {}".format(last_predicted_rewrite)
            if last_response == "":
                last_response = "Unavailable."  # for cast19
            last_dialog_prompt_turns.append("Response: {}".format(last_response))
        else:
            last_dialog_prompt_turns = []
            
        this_turn_prompt = self.build_turn_prompt(this_turn, is_demo=False)    
        last_dialog_prompt_turns.append(this_turn_prompt)
        pre_prompt_components.append("\n".join(last_dialog_prompt_turns))
        
        return "\n\n".join(pre_prompt_components)

    def parse_returned_text(self, text):
        return text.strip()


class RewriteAndResponsePromptor(RewritePrompter):
    def __init__(self) -> None:
        super().__init__()
        self.instruction = "Reformulate the current question into a de-contextualized rewrite under the multi-turn information-seeking dialog context and generate a correct response to the current question."
        self.stop_tokens = ["\n\n", "\nTurn"]
        
    def parse_returned_text(self, text):
        response, rewrite = None, None
        response_pattern = r"Response:\s*(.*)"
        match = re.search(response_pattern, text)
        if match:
            response = match.group(1)    
        else:
            raise ValueError("Response not found in returned text: {}".format(text))
        
        rewrite_pattern = r'^(.*?)\nResponse:' 
        match = re.search(rewrite_pattern, text, re.DOTALL)
        if match:
            rewrite = match.group(1)
        else:
            raise ValueError("Rewrite not found in returned text: {}".format(text))

        return rewrite, response
    
    
    
class RewriteThenResponsePromptor(RewritePrompter):
    def __init__(self) -> None:
        super().__init__()
        self.instruction = "Generate a correct response to the current question rewrite under the multi-turn information-seeking dialog context."
    
    def build_turn_prompt(self, turn, is_demo, predicted_rewrite=None):
        turn_prompt = []
        turn_prompt.append("Turn: {}".format(turn['turn_id']))
        turn_prompt.append("Question: {}".format(turn['question']))
        if is_demo:
            turn_prompt.append("Rewrite: {}".format(turn['manual_rewrite']))
            turn_prompt.append("Response: {}".format(turn['response']))
        else: # for test
            turn_prompt.append("Rewrite: {}".format(predicted_rewrite))
            turn_prompt.append("Response: ")

        return "\n".join(turn_prompt)

    def build_this_turn_prompt_for_prediction(self, pre_prompt, this_turn, this_turn_predicted_rewrite, last_response):
        pre_prompt_components = pre_prompt.split("\n\n")

        # update the last turn of the last dialog's info in the prompt
        if last_response is not None:
            last_dialog_prompt = pre_prompt_components[-1]
            pre_prompt_components.pop()
            last_dialog_prompt_turns = last_dialog_prompt.split('\n')
            if last_response == "":
                last_response = "Unavailable."  # for cast19
            last_dialog_prompt_turns[-1] = "Response: {}".format(last_response)
        else:
            last_dialog_prompt_turns = []
            
        this_turn_prompt = self.build_turn_prompt(this_turn, is_demo=False, predicted_rewrite=this_turn_predicted_rewrite)    
        last_dialog_prompt_turns.append(this_turn_prompt)
        pre_prompt_components.append("\n".join(last_dialog_prompt_turns))
        
        return "\n\n".join(pre_prompt_components)

    def parse_returned_text(self, text):
        return text.strip()
    
    


class CoTRewritePromptor(RewritePrompter):
    def __init__(self) -> None:
        super().__init__()
    
    def build_turn_prompt(self, turn, is_demo):
        turn_prompt = []
        turn_prompt.append("Turn: {}".format(turn['turn_id']))
        turn_prompt.append("Question: {}".format(turn['question']))
        if is_demo:
            output = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
            turn_prompt.append("Rewrite: {}".format(output))
            turn_prompt.append("Response: {}".format(turn['response']))
        else: # for test
            turn_prompt.append("Rewrite: ")
        
        return "\n".join(turn_prompt)

    def parse_returned_text(self, text):
        cot, rewrite = None, None
        cot_pattern = r'^(.*?) So the question should be rewritten as:'
        match = re.search(cot_pattern, text, re.DOTALL)
        if match:
            cot = match.group(1)    
        else:
            raise ValueError("Chain-of-Thought not found in returned text: {}".format(text))
        
        rewrite_pattern = r"So the question should be rewritten as:\s*(.*)"
        match = re.search(rewrite_pattern, text, re.DOTALL)
        if match:
            rewrite = match.group(1)
        else:
            raise ValueError("Rewrite not found in returned text: {}".format(text))

        rewrite_part_text = text
        return rewrite, cot, rewrite_part_text
    
    


class CoTRewriteAndResponsePromptor(RewriteAndResponsePromptor):
    def __init__(self) -> None:
        super().__init__()

    def build_turn_prompt(self, turn, is_demo):
        turn_prompt = []
        turn_prompt.append("Turn: {}".format(turn['turn_id']))
        turn_prompt.append("Question: {}".format(turn['question']))
        if is_demo:
            output = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
            turn_prompt.append("Rewrite: {}".format(output))
            turn_prompt.append("Response: {}".format(turn['response']))
        else: # for test
            turn_prompt.append("Rewrite: ")
        
        return "\n".join(turn_prompt)
    
    
    def parse_returned_text(self, text):
        # text: "cot, so the question should be rewritten as: rewrite\nResponse: response\n..."
        cot, rewrite, response = None, None, None
        cot_pattern = r'^(.*?) So the question should be rewritten as:'
        match = re.search(cot_pattern, text, re.DOTALL)
        if match:
            cot = match.group(1)    
        else:
            raise ValueError("Chain-of-Thought not found in returned text: {}".format(text))

        rewrite_pattern = r'So the question should be rewritten as: (.*?)\nResponse:'
        match = re.search(rewrite_pattern, text, re.DOTALL)
        if match:
            rewrite = match.group(1)
        else:
            raise ValueError("Rewrite not found in returned text: {}".format(text))

        response_pattern = r"Response:\s*(.*)"
        match = re.search(response_pattern, text)
        if match:
            response = match.group(1)    
        else:
            raise ValueError("Response not found in returned text: {}".format(text))

        rewrite_part_text_pattern = r'^(.*?)\nResponse:' 
        match = re.search(rewrite_part_text_pattern, text, re.DOTALL)
        if match:
            rewrite_part_text = match.group(1)
        else:
            raise ValueError("rewrite_part_text not found in returned text: {}".format(text))


        return rewrite, response, cot, rewrite_part_text

    