"""
two_stage_pipeline.py

This script processes a folder of PDF files, applies a two-stage LLM pipeline to extract structured epidemiological parameters from each document, and saves the results to a CSV file.

Integrated Components:
- `ParameterExtractor` class (defined in this script) performs the core double-layered LLM-based extraction:
    1. Extracts text from PDFs using Azure Document Intelligence.
    2. Sends the full text to GPT with a system prompt and target parameters.
    3. Refines the output via a second GPT call with a refining prompt.
- Configuration files in the `config/` directory define extraction parameters and LLM prompts.

Typical usage:
    python two_stage_pipeline.py --folder path/to/folder --output results.csv --verbose

Dependencies:
- Azure Document Intelligence (`TextExtractor`)
- Azure OpenAI (`ask_GPT`)
- pandas
- json
- argparse
- utils.utils.load_config
"""

from utils.utils import load_config
from LLM_interaction.gpt_client import ask_GPT
from text_extractor.docint import TextExtractor
import argparse
import os
import json
import pandas as pd

class ParameterExtractor:
    def __init__(self, file_path: str, parameters: list[str], sys_prompt: str, refine_prompt: str):
        """Pipeline for parameter extraction of single PDF file."""
        self.file_path = file_path
        self.parameters = parameters
        self.sys_prompt = sys_prompt
        self.refine_prompt = refine_prompt
        self.first_response = None
        self.refined_response = None
        self.extraction_performed = False

    def extract_text(self) -> None:
        """Extracts text from the PDF file using the TextExtractor class."""
        self.extractor = TextExtractor()
        self.extractor.extract_text(self.file_path)
        self.article_text = self.extractor.full_text
        
    def first_query(self) -> None:
        """Sends the initial request to ChatGPT and stores the response."""
        if not hasattr(self, 'article_text'):
            raise RuntimeError("Please run extract_text() before first_query().")
        
        self.first_prompt = [{"role": "system", "content": self.sys_prompt},
                             {"role": "user", "content": f"This is the article text:\n{self.article_text}\n\n"},
                             {"role": "user", "content": f"These are the requested parameters:\n{self.parameters}"}]
        self.first_response = ask_GPT(prompt=self.first_prompt)
    
    def refine_query(self) -> None:
        """Refines and formats previous ChatGPT responses."""
        if self.first_response is None:
            raise RuntimeError("Please run first_query() before refine_query().")
        self.second_prompt = [{"role": "system", "content": self.refine_prompt},
                              {"role": "user", "content": f"This is the text:\n{self.first_response}\n\n"},
                              {"role": "user", "content": f"These are the requested parameters:\n{self.parameters}"}]
        self.refined_response = ask_GPT(prompt=self.second_prompt)

    def get_parameters(self) -> None:
        """Returns the refined response."""
        self.extract_text()
        self.first_query()
        self.refine_query()
        self.extraction_performed = True
            
    def export_parameters(self, output_dir: str = "output") -> None:
        """(FOR TESTING PURPOSES) Exports the LLM's responses to a .txt file. If an extraction has not been performed, raises a ValueError."""
        if not self.extraction_performed: raise ValueError('An extraction has not yet been performed.')

        os.mkdir(output_dir)
        first_response_path = os.path.join(output_dir, "first_response.txt")
        refined_response_path = os.path.join(output_dir, "refined_response.txt")
        if os.path.exists(first_response_path) or os.path.exists(refined_response_path):
            print(f"Warning: path already exists. Overwriting.")
        # Export the text
        with open(first_response_path, "w") as text_file:
            text_file.write(self.first_response)
        with open(refined_response_path, "w") as text_file:
            text_file.write(self.refined_response)


def main(folder_path: str, output_path: str = "output", get_explanations: bool = True, verbose: bool = False, ) -> None:
    """
    Processes all PDF files in a folder, extracts specified parameters using GPT, 
    and saves the results to a CSV file.
    """
    prompts: dict = load_config("config/prompts.json")
    sys_prompt: str = prompts["sys_prompt"]
    refine_prompt: str = prompts["refine_prompt"]
    parameters: list[str] = load_config("config/parameters.json")["parameters"]

    data: list[dict] = []
    titles: list[str] = []
    if get_explanations:
        explanations: list[str] = []
    n: int = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            extractor = ParameterExtractor(file_path, parameters, sys_prompt, refine_prompt)
            extractor.get_parameters()
            found_parameters: dict = json.loads(extractor.refined_response)
            # Add results to list
            data.append(found_parameters)
            # Label with file name
            titles.append(os.path.basename(file_path))
            # Add explanations if requested
            if get_explanations:
                explanations.append(extractor.first_response)
            n+=1
            if verbose:
                print(f"File {n} processed.")
        else:
            next
    if verbose:
        print(f"{n} files processed.")

    if get_explanations:
        explanations_path = os.path.join(output_path, "explanations.txt")
        with open(explanations_path, "w") as file:
            for exp in explanations:
                file.write(f"{exp}\n\n")

    df =  pd.DataFrame(data)
    df["Paper"] = titles
    output_file = os.path.join(output_path, "twostage_results.csv")
    df.to_csv(output_file, index=False)
    
    return(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the two-stage extraction pipeline on a folder of PDFs.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing PDF files.")
    parser.add_argument("--output", default="output.csv", help="Path to save the output CSV file.")
    parser.add_argument("--explanations", action="store_true", help="Enable storage of explanations.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()
    
    main(folder_path=args.folder, output_path=args.output, get_explanations=args.explanations, verbose=args.verbose)
