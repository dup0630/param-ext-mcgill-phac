"""
Pipeline for extracting epidemiological parameters from PDFs using:
- Azure Document Intelligence (for OCR/text extraction)
- GPT (for parameter extraction and explanation)
Saves extracted results to a central CSV and tracks performance across iterations.
The main function takes as input the path to the location where these files are stored.

CURRENT CONFIGURATION:
This script is configured to extract the **Case Fatality Rate (CFR)** for **measles** only.

To generalize for other parameters or diseases, update the following:

1. `parameters` list in the `main()` function  
   ➤ Currently: ["Case Fatality Rate (CFR)"]  
   ➤ Change to: your new target parameter name(s) (must match column names in your input CSVs)

2. `pipeline_testing.csv` (ground truth values file)  
   ➤ Column "True CFR" must be renamed or replaced to match your new parameter  
   ➤ Example: For "Length of Hospital Stay", add a column "True Length of Stay" or similar

3. Output CSV file `CFR_measles.csv`  
   ➤ This is the file where results are saved  
   ➤ Recommended: create a new file, e.g., `length_of_stay.csv`, and change the name in:
       - `load_data_and_setup(csv_path=...)`
       - `update_csv_with_results(df, csv_file, result)`

OPTIONAL:
- You may add a mapping from parameter → expected truth column name and → result output file.
- This will allow switching parameters without modifying the script each time.

This script is designed for experimentation with prompt refinement and evaluation before integrating prompts into the full epidemiological extraction pipeline.
"""
import os
import pandas as pd
import re
import argparse
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

def initialize_clients():
    """Load environment variables and initialize OpenAI and Azure Document Intelligence clients."""
    load_dotenv()

    #OpenAI setup
    api_key = os.getenv("OPENAI_KEY")
    endpoint = os.getenv("OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_VERSION", "2024-02-01")  # fallback if not set
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    #Document Intelligence setup
    docint_key = os.getenv("DOCINT_KEY")
    docint_endpoint = os.getenv("DOCINT_ENDPOINT")
    docint_client = DocumentAnalysisClient(
        endpoint=docint_endpoint,
        credential=AzureKeyCredential(docint_key)
    )

    return client, docint_client


def load_data_and_setup(csv_path="CFR_measles.csv", excel_path="pipeline_testing.csv", cache_dir="cached_texts"):
    """Load the annotation CSV and the ground truth Excel file."""
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    paper_df = pd.read_csv(excel_path)
    print(f"CSV data loaded successfully. {len(df)} rows found.")

    required_columns = ["Prompt", "Model Name", "Parameter Name", "Paper Number", 
                        "Extracted Parameter", "True Parameter", "Success/Fail", "Confusion", "Iteration"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
            
    #Determine current iteration number for this run
    if "Iteration" in df.columns and not df["Iteration"].isna().all():
        current_iteration = int(df["Iteration"].max()) + 1
    else:
        current_iteration = 1
    
    print(f"\nStarting Iteration {current_iteration}...\n")
    
    #set up cache directory
    os.makedirs(cache_dir, exist_ok=True)

    return df, paper_df, current_iteration, cache_dir


def get_sorted_paper_numbers(paper_df):
    """
    Extract numeric paper identifiers from the 'PDF' column of paper_df
    and return them sorted numerically.
    """

    paper_list = paper_df["PDF"].dropna().astype(str).unique().tolist()
    paper_numbers = sorted(paper_list, key=extract_numeric)
    print(f"Sorted Paper Numbers: {paper_numbers}")
    return paper_numbers

def extract_numeric(paper):
    """
    Extract the numeric ID from paper names (e.g., '12.pdf' → 12).
    Used for sorting and comparison.
    """
    match = re.search(r'\d+', paper)
    return int(match.group()) if match else float('inf')


def get_next_paper(df, paper_numbers):

    last_paper_tested = df["Paper Number"].dropna().astype(str).iloc[-1] if not df.empty else None
    next_paper = None
    if last_paper_tested:
        last_numeric = extract_numeric(last_paper_tested)
        for paper in paper_numbers:
            if extract_numeric(paper) > last_numeric:
                next_paper = paper
                break
    return next_paper if next_paper else paper_numbers[0]




def extract_text_from_pdf(pdf_path, paper_number, cache_dir, document_analysis_client):
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
    with open(pdf_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-document", document=f)
    result = poller.result()
    text = "\n".join([line.content for page in result.pages for line in page.lines])
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text



def generate_improved_prompt(parameter, previous_prompts, client):
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
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating improved prompt for {parameter}: {e}")
        return "Not Found"


def extract_parameters(pdf_text, prompt, parameter, client):
    """
    Use GPT to extract the value of a given parameter from the document text.
    The full prompt includes:
    - A long-form retrieval instruction block
    - The improved prompt generated earlier
    - The truncated document text (first 16,000 characters)
    
    Returns raw GPT output (including explanation and value).
    """

    retrieval_instructions = """
        You are a **data retrieval assistant for medical research papers**, specialized in extracting epidemiological parameters from full-text papers. 
        
        #### User Input:
        - A full-text medical research paper.
        - A list of epidemiological parameters of interest (may include definitions or explanations).
        
        #### Your Task:
        - Identify and extract the value of each parameter from the paper.
        - Provide a **brief explanation** of where you found it and why it is the appropriate value.
        - If the paper reports **multiple values** for a parameter due to population subgroups (e.g., hospitalized vs. non-hospitalized, age groups), report each value along with a short note on its context (e.g., “Hospitalized patients”, “Children under 5”).
        - If the value must be **computed or inferred**, do so clearly and explain how you derived it.
        
        #### Additional Instructions for CFR Priority:
        When extracting a Case Fatality Rate (CFR), apply the following priority order:
        
        1. If a **CFR for hospitalized measles patients** is explicitly stated, extract it.
        2. If **deaths among hospitalized measles patients and total hospitalized cases** are provided, compute the CFR.
        3. If a **general measles CFR** is explicitly stated (without hospitalization filter), extract it only if no hospitalized CFR is available.
        4. If only **total deaths and total measles cases** are given, compute the CFR.
        5. If no numerical CFR is provided, but the text says **no measles-related deaths occurred** or infers this information in some way, return CFR = 0% and explain that this is implied.
        6. If no relevant information is available, return **"Not found."**
        
        Do not extract CFRs related to diseases other than measles (e.g., tetanus, shigella).
        
        #### Disease-Specific and Contextual Restrictions:
        - Only extract a CFR if it clearly relates to **measles**, either generally or for a specific subgroup (e.g., hospitalized patients, children, etc.).
        - Do not extract CFRs associated with **other diseases** (e.g., Shigella, pneumonia).
        - If the paper states that **no measles-related deaths occurred** and measles cases were tracked, you may infer CFR = 0% and explain.
        - If **measles cases were tracked but no CFR or death info is given**, return **"Not found."**
        - If there is **no evidence of measles cases at all**, return **"Not found."**
        - If sections of the document are in other languages (e.g., French), attempt to interpret them when relevant.
        
        #### Guidelines:
        1. **If a parameter’s value is in a different format than expected** (e.g., percentage vs. decimal), return it **as found** and briefly explain why it corresponds to the requested parameter.
        2. **If the exact value is not found but related values allow deduction, computation, or approximation**, return those values with an explanation of their relevance.
        3. **If no relevant value is found, return "Not found."**
        4. **Keep responses concise**—each parameter’s response should be **no more than a few lines**.

    """

    # Construct the full prompt
    full_prompt = f"""{retrieval_instructions}

    **Parameter to Extract:** {parameter}
    {prompt}  # <-- This is the improved prompt

    **Document Text:**
    {pdf_text[:16000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0
    )

    raw_response = response.choices[0].message.content
    print(f"\n**ChatGPT Response for {parameter}:**\n{raw_response}\n")

    # Return the extracted response as-is
    return raw_response if raw_response else "NA"



def update_csv_with_results(df, csv_file, result):
    """
    Append the result (containing all metadata and extracted info)
    to the DataFrame and save the updated CSV to disk.
    """
    new_row = pd.DataFrame([result])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Entry added to {csv_file}")
    return df



def main(directory):
    """
    For each parameter and each paper:
    - Generate an improved prompt
    - Load the PDF and extract text (from cache or API)
    - Use GPT to extract the parameter value
    - Compare to ground truth (prompt user for success/failure)
    - Log results to the CSV
    """
    client, document_analysis_client = initialize_clients()
    df, paper_df, current_iteration, cache_dir = load_data_and_setup()


    paper_numbers = get_sorted_paper_numbers(paper_df)

    parameters = ["Case Fatality Rate (CFR)"]
    
    for param in parameters:
      previous_prompts = "\n".join(df[df["Parameter Name"] == param]["Prompt"].dropna().tolist())
      improved_prompt = generate_improved_prompt(param, previous_prompts, client)
  
      for paper in paper_numbers:
          print(f"\nNow processing paper: {paper}")
          target_file = f"{paper}.pdf"
          pdf_path = os.path.join(directory, target_file)
  
          if not os.path.exists(pdf_path):
              print(f"No PDF found for paper {paper}. Skipping.")
              continue
  
          pdf_text = extract_text_from_pdf(pdf_path, paper, cache_dir, document_analysis_client)
          extracted_value = extract_parameters(pdf_text, improved_prompt, param, client)
  
          # Lookup true parameter
          true_param_lookup = paper_df.loc[paper_df["PDF"] == int(paper), "True CFR"]
          true_param = true_param_lookup.iloc[0] if not true_param_lookup.empty else "NA"
          print(f"The True parameter from Excel for '{param}' (paper {paper}): {true_param}")
  
          success_fail = input(f"Was it successful? (Success/Fail): ").strip()
          confusion_level = input("Is it a TP/TN/FP/FN: ").strip()
          result = {
              "Prompt": improved_prompt,
              "Model Name": "gpt-4o-mini",
              "Parameter Name": param,
              "Paper Number": paper,
              "Extracted Parameter": extracted_value,
              "True Parameter": true_param,
              "Success/Fail": success_fail,
              "Confusion": confusion_level,
              "Iteration": current_iteration
          }
          df = update_csv_with_results(df, "CFR_measles.csv", result)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prompt refinement pipeline.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing PDF files.")
    args = parser.parse_args()

    main(directory=args.folder)


