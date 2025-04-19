"""
prompt_refiner_manual.py

This module implements an interactive pipeline for manually guided prompt refinement and parameter extraction 
from epidemiological research papers using GPT models. The process is tailored for iterative experimentation 
and performance tracking.
Saves extracted results to a central CSV and tracks performance across iterations.

- Loads previously extracted results and true parameter values for a given set of papers.
- Generates improved prompts for parameter extraction based on historical prompt performance.
- Uses Azure Document Intelligence to extract and cache text from PDFs.
- Queries GPT to extract specific parameter values from each paper using the refined prompt.
- Prompts the user to manually assess the extraction outcome and label it (Success/Fail, TP/TN/FP/FN).
- Logs all results, metadata, and annotations into a cumulative CSV for iterative tracking.

This script is designed for experimentation with prompt refinement and evaluation before integrating prompts into the full epidemiological extraction pipeline.

"""
import os
import pandas as pd
from text_extractor.docint import TextExtractor
from LLM_interaction.gpt_client import ask_GPT
from utils.utils import load_config


def load_data_and_setup(output_path: str, true_param_path: str, cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, int, str]:
    """
    Load the annotation and true parameters CSV files.
    """

    results_df = pd.read_csv(output_path)
    true_param_df = pd.read_csv(true_param_path)

    required_columns = ["Prompt", "Model Name", "Parameter Name", "Paper Number", 
                        "Extracted Parameter", "True Parameter", "Success/Fail", "Confusion", "Iteration"]
    for col in required_columns:
        if col not in results_df.columns:
            results_df[col] = ""
            
    #Determine current iteration number for this run
    if "Iteration" in results_df.columns and not results_df["Iteration"].isna().all():
        current_iteration = int(results_df["Iteration"].max()) + 1
    else:
        current_iteration = 1
    
    print(f"\nStarting Iteration {current_iteration}...\n")
    
    #set up cache directory
    os.makedirs(cache_dir, exist_ok=True)

    return results_df, true_param_df, current_iteration, cache_dir


def extract_text_from_pdf(pdf_path: str, paper_number: str, cache_dir: str) -> str:
    """
    Extract text from a PDF using Azure Document Intelligence.
    If the text was previously extracted and saved, reuse it from cache.
    """
    cache_file = os.path.join(cache_dir, f"{paper_number}.txt")
    if os.path.exists(cache_file):
        print(f"Using cached text for {pdf_path} (no cost!).")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    print(f"Using Azure Document Intelligence for {pdf_path} (this costs money :( )")

    extractor = TextExtractor(output_dir="test_output")
    extractor.extract_text(pdf_path, verbose=True)
    extractor.export_text(output_name="test.txt")
    text = extractor.full_text

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def generate_improved_prompt(parameter: str, previous_prompts: str) -> str:
    """
    Use GPT to generate a refined prompt for extracting a specific parameter,
    based on the history of previously used prompts stored in the CSV.
    """
    print(f"Generating improved prompt for {parameter}...")
    prompt_text = (
        "You are an AI assistant extracting specific **epidemiological parameters** from research papers.\n\n"
        "### Task:\n"
        f"Your job is to extract only the **{parameter}** from the given research paper text.\n\n"
        "### How to Improve Extraction:\n"
        "Below are previous prompts used for extracting this parameter. Improve upon them to maximize accuracy:\n\n"
        f"{previous_prompts}\n\n"
        "### Response Guidelines:\n"
        "- **Your response must be a single improved prompt** for extracting the parameter.\n"
        "- **DO NOT include explanations or introductions. Only return the improved prompt.**"
    )
    prompt = [{"role": "user", "content": prompt_text}]
    try:
        response = ask_GPT(prompt=prompt)
        return response
    except Exception as e:
        print(f"Error generating improved prompt for {parameter}: {e}")
        return "Not Found"


def extract_parameters(pdf_text: str, prompt: str, parameter: str) -> str:
    """
    Use GPT to extract the value of a given parameter from the document text.
    The full prompt includes:
    - A long-form retrieval instruction block
    - The improved prompt generated earlier
    - The truncated document text (first 16,000 characters)
    
    Returns raw GPT output (including explanation and value).
    """

    with open("config/refiner_prompt.txt", "r") as prompt_file:
        retrieval_instructions = prompt_file.read

    # Construct the full prompt
    full_prompt = f"""{retrieval_instructions}

    **Parameter to Extract:** {parameter}
    {prompt}  # <-- This is the improved prompt

    **Document Text:**
    {pdf_text[:16000]}
    """
    raw_response = ask_GPT(prompt=full_prompt)
    print(f"\n**ChatGPT Response for {parameter}:**\n{raw_response}\n")

    # Return the extracted response as-is
    return raw_response if raw_response else "NA"


def update_csv_with_results(df: pd.DataFrame, csv_file: str, result: dict) -> pd.DataFrame:
    """
    Append the result (containing all metadata and extracted info)
    to the DataFrame and save the updated CSV to disk.
    """
    new_row = pd.DataFrame([result])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Entry added to {csv_file}")
    return df


def main(directory: str, results_path: str , true_param_path: str, cache_dir: str = "cached_texts") -> None:
    """
    For each parameter and each paper:
    - Generate an improved prompt
    - Load the PDF and extract text (from cache or API)
    - Use GPT to extract the parameter value
    - Compare to ground truth (prompt user for success/failure)
    - Log results to the CSV
    """
    results_df, true_param_df, current_iteration, cache_dir = load_data_and_setup(results_path, true_param_path, cache_dir)

    parameters_dict: dict = load_config("config/refiner_parameters.json")
    parameter_colnames = list(parameters_dict.keys())
    parameters = list(parameters_dict.values())  
    

    for i, param in enumerate(parameters):
        param_colname = parameter_colnames[i]
        previous_prompts = "\n".join(results_df[results_df["Parameter Name"] == param]["Prompt"].dropna().tolist())
        improved_prompt = generate_improved_prompt(param, previous_prompts)
    
        for index, row in true_param_df.iterrows():
            true_param = row[param_colname]
            filename = row["PDF"]
            pdf_path = os.path.join(directory, f"{filename},pdf")
            if not os.path.exists(pdf_path):
                print(f"No PDF found for paper {filename}. Skipping.")
                continue

            pdf_text = extract_text_from_pdf(pdf_path, filename, cache_dir)
            extracted_value = extract_parameters(pdf_text, improved_prompt, param)

            print(f"The True parameter for '{param}' (paper {filename}): {true_param}")
    
            success_fail = input(f"Was it successful? (Success/Fail): ").strip()
            confusion_level = input("Is it a TP/TN/FP/FN: ").strip()
            result = {
                "Prompt": improved_prompt,
                "Model Name": "gpt-4o-mini",
                "Parameter Name": param,
                "Paper Number": filename,
                "Extracted Parameter": extracted_value,
                "True Parameter": true_param,
                "Success/Fail": success_fail,
                "Confusion": confusion_level,
                "Iteration": current_iteration
            }
            results_df = update_csv_with_results(results_df, results_path, result)
  
if __name__ == "__main__":
    pdf_dir = "crf_validation/test_papers"
    results_path = "cfr_validation/CFR_measles.csv"
    true_cfr = "crf_validation/true_parameters.csv"

    main(directory=pdf_dir, results_path=results_path, true_param_path=true_cfr)


