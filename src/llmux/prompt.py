import os

class Prompt:
    def __init__(self, prompt, **kwargs):
        if os.path.exists("./prompts/" + prompt):
            with open("./prompts/" + prompt, "r") as f:
                self.raw_prompt = f.read()
        elif os.path.exists(prompt):
            with open(prompt, "r") as f:
                self.raw_prompt = f.read()
        else:
            self.raw_prompt = prompt
        
        self.raw_prompt = self.prep_prompt(self.raw_prompt, **kwargs)

    def __str__(self) -> str:
        return self.raw_prompt

    def prep_prompt(self, prompt, **kwargs):
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", value)
        return prompt

    def get(self, **kwargs):
        return self.prep_prompt(self.raw_prompt, **kwargs)
    
