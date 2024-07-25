import json

def load_prompt(query, prompt_type):
    prompts = json.load(open("bnf/prompts.json"))
    prompt = prompts[prompt_type]
    return prompt.replace("{{REQUETE}}",query)


#print(load_prompt("1er guerre mondiale","zero-shot"))


