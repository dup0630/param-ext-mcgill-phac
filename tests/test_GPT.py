from LLM_interaction.gpt_client import ask_GPT
from utils.utils import load_config

prompts = load_config("config/prompts.json")
sys_prompt = prompts["sys_prompt"]
refine_prompt = prompts["refine_prompt"]
parameters = load_config("config/parameters.json")["parameters"]

response = ask_GPT(
    prompt=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"This is the article text:\n There were 5 obese patients and 10 non-obese patiens. No complications occurred.\n\n"},
        {"role": "user", "content": f"These are the requested parameters:\n{parameters}"}
    ]
)

print(response)