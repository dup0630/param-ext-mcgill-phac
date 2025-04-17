from two_stage_pipeline import ParameterExtractor
from utils.utils import load_config

prompts = load_config("config/prompts.json")
sys_prompt = prompts["sys_prompt"]
refine_prompt = prompts["refine_prompt"]
parameters = load_config("config/parameters.json")["parameters"]

file_path = input("Enter file path: ")

extractor = ParameterExtractor(file_path, parameters, sys_prompt, refine_prompt)
extractor.get_parameters()
found_parameters = extractor.refined_response
extractor.export_parameters(output_dir="output")
print(found_parameters)