"""
legacy_cfr_extraction.py

Legacy script for extracting Hospitalized Case Fatality Rate (CFR) from measles-related papers.
Supports both sampled evaluation (with true CFR labels) and full-batch processing.

Usage:
    python legacy_cfr_extraction.py --mode sampled
    python legacy_cfr_extraction.py --mode all

This script:
- Uses CFR-specific extraction and formatting prompts
- Sends two GPT queries per paper: one for raw extraction, one for structured field output
- Saves results to an Excel file with two sheets
"""

import os
import csv
import argparse
import pandas as pd
from openai import AzureOpenAI
import re
from dotenv import load_dotenv
from utils.utils import load_config

# ---------------- Configuration ---------------- #
load_dotenv()
key = os.getenv("OPENAI_KEY")
endpoint = os.getenv("OPENAI_ENDPOINT")
version = os.getenv("OPENAI_VERSION")
if not key or not endpoint:
    raise ValueError("OPENAI_KEY and/or OPENAI_ENDPOINT not set in environment variables.")
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version=version
)

papers_directory = "cfr_validation/paper_texts"
excel_path_sampled = "cfr_validation/sampledstdFormatCFR.xlsx"
excel_path_all = "cfr_validation/ALLstdFormatCFR.xlsx"
true_parameters_path = "cfr_validation/true_parameters.csv"

# --------------- Prompts ---------------- #
extraction_prompt = """[same as before, truncated here for brevity]"""
standard_extraction_prompt = """[same as before, truncated here for brevity]"""

# --------------- Helpers ---------------- #
def read_csv_as_string(csv_path: str) -> str:
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return "\n".join(", ".join(row) for row in csv.reader(f))
    except Exception as e:
        print(f"Error reading csv {csv_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def parse_standard_text(text: str) -> dict:
    return {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
            for line in text.splitlines() if ":" in line}

def extract_overall_hosp_cfr(raw_text: str) -> str:
    match = re.search(r"Overall\s+Hospitalized\s+CFR\s*[:=]\s*\**([0-9.]+)\**", raw_text, re.IGNORECASE)
    return match.group(1) if match else ""

def extract_int(value) -> int:
    value = str(value) if value is not None else ""
    cleaned = re.sub(r"[^\d]", "", value)
    return int(cleaned) if cleaned else None

def calculate_cfr(numerator_str, denominator_str):
    num = extract_int(numerator_str)
    den = extract_int(denominator_str)
    return num / den if num is not None and den and den != 0 else ""

# --------------- Main Script ---------------- #
def run_extraction(mode: str):
    is_sampled = (mode == "sampled")
    excel_path = excel_path_sampled if is_sampled else excel_path_all

    # Collect paper IDs
    if is_sampled:
        papers_df = pd.read_csv(true_parameters_path)
        papers_df['PDF'] = papers_df['PDF'].astype(str)
        paper_ids = papers_df['PDF'].tolist()
        true_cfr_lookup = dict(zip(papers_df["PDF"], papers_df["TrueCFR"]))
    else:
        paper_ids = [entry for entry in os.listdir(papers_directory)
                     if os.path.isdir(os.path.join(papers_directory, entry))]
        true_cfr_lookup = {}

    raw_output_data = []
    standard_output_data = []

    for paper_id in paper_ids:
        folder = os.path.join(papers_directory, paper_id)
        txt_path = os.path.join(folder, f"{paper_id}.txt")
        csv_path = os.path.join(folder, f"{paper_id}.csv")

        pdf_text = read_text_file(txt_path)
        csv_data = read_csv_as_string(csv_path) if os.path.exists(csv_path) else ""
        if not pdf_text:
            print(f"Skipping {paper_id}: no text content found.")
            continue

        # --- Raw Extraction --- #
        raw_prompt = f"""
{extraction_prompt}
Table Data:
{csv_data[:10000]}
Document Text:
{pdf_text[:25000]}
"""
        print(f"\n[Raw Extraction] {paper_id}")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": raw_prompt}],
                temperature=0
            )
            raw_response = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing {paper_id} (raw): {e}")
            raw_response = "Error"

        raw_output = {
            "Papers": paper_id,
            "Extracted Response": raw_response,
            "overall CFR": extract_overall_hosp_cfr(raw_response)
        }
        if is_sampled:
            raw_output["TrueCFR"] = true_cfr_lookup.get(paper_id, "")
        raw_output_data.append(raw_output)

        # --- Standard Extraction --- #
        std_prompt = f"""
PDF: {paper_id}
{standard_extraction_prompt}
Table Data:
{csv_data[:10000]}
Document Text:
{pdf_text[:25000]}
"""
        print(f"[Standard Extraction] {paper_id}")
        try:
            response_std = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": std_prompt}],
                temperature=0
            )
            std_response = response_std.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing {paper_id} (standard): {e}")
            std_response = "Error"

        parsed_data = parse_standard_text(std_response)
        parsed_data["PDF"] = parsed_data.get("PDF", paper_id)
        standard_output_data.append(parsed_data)

    # --- Save Results --- #
    df_raw = pd.DataFrame(raw_output_data)
    standard_columns = [
        "PDF", "cases confirmed", "cases suspected", "# symptomatic cases", "# hospitalized", "# deaths",
        "Sample size - number of observations", "Sample size - number of studies", "Age_min", "Age_max",
        "Parameter Value", "Parameter range - lower value", "Parameter range - upper value",
        "Statistical approach", "Numerator", "Denominator"
    ]
    df_std = pd.DataFrame(standard_output_data)
    for col in standard_columns:
        if col not in df_std.columns:
            df_std[col] = ""
    df_std["calculated CFR"] = df_std.apply(
        lambda row: calculate_cfr(row.get("Numerator", ""), row.get("Denominator", "")), axis=1)
    df_std = df_std[standard_columns + ["calculated CFR"]]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="raw response", index=False)
        df_std.to_excel(writer, sheet_name="standard format", index=False)

    print(f"\n✅ Completed. Results saved to: {excel_path}")


# --------------- CLI Entry ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run legacy CFR extraction with GPT.")
    parser.add_argument("--mode", choices=["sampled", "all"], required=True,
                        help="Run on 'sampled' set (with labels) or 'all' papers in the folder.")
    args = parser.parse_args()
    run_extraction(mode=args.mode)
