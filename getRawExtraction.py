import os
import csv
import json
import pandas as pd
from openai import AzureOpenAI
from typing import List
import re

# ------------------------ Configuration ------------------------ #
# replace your paths for key and endpoints
with open('/home/cdsi/users/yinggui.li@MAIL.MCGILL.CA/key.txt', 'r') as file:
    api_key = file.read().strip()
with open('/home/cdsi/users/yinggui.li@MAIL.MCGILL.CA/endpoint.txt', 'r') as file:
    endpoint = file.read().strip()
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-02-01"
)

# replace your paths for paper text&csv
papers_directory = "/home/cdsi/users/yinggui.li@MAIL.MCGILL.CA/winter-2025-phac/allPapers"

# replace your path for the output excel file
excel_path = "/home/cdsi/users/yinggui.li@MAIL.MCGILL.CA/winter-2025-phac/sampledstdFormatCFR.xlsx"

# ------------------------ Extraction Prompts ------------------------ #
# Prompt for raw extraction (hospitalized CFR) – UPDATED
extraction_prompt = """
Extract the values for the parameter Hospitalized Case Fatality Rate (CFR) for Measles from the provided document.
Guidelines:
1. Recognize that Case Fatality Rate (CFR) is defined as the proportion of patients who die among cases, and in this context, it should only include those who were formally admitted (hospitalized) due to illness severity.
2. If the document directly provides a percentage value for the Hospitalized CFR and no raw numbers are available, extract that percentage as the overall Hospitalized CFR without further calculation. Otherwise, extract the raw numbers (i.e., number of deaths and total hospitalized cases) used to derive the CFR.
3. Only count deaths that occurred during hospitalization. DO NOT include any death counts from cases where:
   - Hospitalization was refused (e.g., “parents refused hospitalization”),
   - Patients left against medical advice (e.g., “taken away from the ward”),
   - Any events outside the formal hospitalized setting,
   - Where the death is explicitly noted as not fully attributable to measles
   - If the document states that “no deaths during hospitalization” or similar phrasing is present, then assume the number of hospitalized deaths is 0 and calculate the CFR accordingly.
4. There are two types of CFR:
   - CFR general: Pertains to the overall population without hospital admission. Only report this if both the numerator and denominator (i.e., raw numbers) are provided or derivable, and only if hospitalized CFR is also available.
   - CFR hospitalized: Pertains to patients admitted to hospital due to illness severity. Use only data from patients that were admitted and had a conclusive outcome.
5. Studies may report both general and hospitalized CFR or allow both to be inferred. If both are available (or can be inferred), capture them as separate parameters.
6. If multiple subgroups are reported (for example, differences by nutritional status, age, consultation time, etc.), extract:
   - The individual raw numbers and the provided or calculated CFR for each subgroup.
   - Additionally, calculate an overall Hospitalized CFR as the sum of all subgroup deaths divided by the sum of all subgroup hospitalized cases.
7. Be meticulous in calculations:
   - Always recalculate the CFR using the formula: CFR (%) = (Total Hospitalized Deaths / Total Hospitalized Cases) × 100.
   - Round the calculated CFR to two decimal places.
   - Ensure the reported percentages match this calculation.
8. Handle variations in text: Recognize variations in phrases like “all cases recovered”, “no deaths reported”, “no deaths related with the outbreak”, and extract accordingly.
9. If a table is provided (Table Data), and if it contains a row with a “Total” or clearly summative numbers (e.g., 11,076 hospitalized cases and 2274 deaths), use these numbers for calculating the overall Hospitalized CFR.
10. If no value is found, or if the data do not include both the raw number of cases and deaths (or if they are from averaged long-term data rather than annual data), return "NA".

Extract the values following these guidelines.
Lastly, after completing your extraction, please provide a final summary line in the following exact format:
Overall Hospitalized CFR: <value>
where <value> is the overall Hospitalized CFR extracted from the study (or computed as described above).
"""

# Prompt for the standard extraction – UPDATED
standard_extraction_prompt = """
For the purpose of this extraction, Hospitalized CFR is defined as the case fatality rate among patients admitted to the hospital due to illness severity. (Studies that only include cases from peripheral facilities such as outpatient clinics or emergency room visits should not be considered for hospitalized CFR.)
Extract the following details for Hospitalized CFR from the document and format the response as plain text.
For missing values, leave them blank. (Note: '#' means the number of)
Separate multiple reports by a blank line.
- PDF: <value>
- cases confirmed: <value>
- cases suspected: <value>
- # symptomatic cases: <value>
- # hospitalized: <value>
- # deaths: <value>
- Sample size - number of observations: <value>
- Sample size - number of studies: <value>
- Age_min: <value>
- Age_max: <value>
- Parameter Value: <value>
- Parameter range - lower value: <value>
- Parameter range - upper value: <value>
- Statistical approach: <value>
- Numerator: <value>
- Denominator: <value>

Tables and Document Text:
"""

# ------------------------ Selected Papers and their true parameters ------------------------ #
data = {
    'PDF': [75, 88, 98, 104, 213, 242, 511, 554, 584, 585, 1544, 1846, 3083, 3561, 3562, 7471, 2146, 1797, 640, 409, 347, 558, 2209, 1154, 6300, 1754, 275, 664, 1910, 531, 1061, 365, 1641, 469, 1734, 1053, 1287, 908, 2294, 1280, 1159, 395, 817, 1792, 1789],
    'True CFR': [None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.23, 0.25, 0.31, 3.14, 3.23, 3.85, 4.88, 5.72, 6.25, 7.93, 8.64, 9.23, 10, 10.84, 11.48, 12.8, 12.9, 14.2, 14.76, 15.95, 17.86, 18.31, 20.1, 20.53, 26.02, 30.44, 33.03]
}

papers_df = pd.DataFrame(data)
papers_df['PDF'] = papers_df['PDF'].astype(str)

# ------------------------ Helper Functions ------------------------ #

def read_csv_as_string(csv_path: str) -> str:
    lines = []
    try:
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(", ".join(row))
        return "\n".join(lines)
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
    """
    Parse the standard extraction text into a dictionary.
    Each line should have the format 'FieldName: <value>'.
    If no value is provided, that key will be set to an empty string.
    """
    d = {}
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(":", 1)
            d[key.strip()] = value.strip()
    return d
  
def extract_overall_hosp_cfr(raw_text: str) -> str:
    """
    Extracts the overall Hospitalized CFR value from the raw GPT output.
    Looks for the pattern "Overall Hospitalized CFR: <value>".
    If found, returns the extracted value as a string. Otherwise, returns an empty string.
    """
    match = re.search(r"Overall\s+Hospitalized\s+CFR\s*[:=]\s*\**([0-9.]+)\**", raw_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""
  
def extract_int(value) -> int:
    """
    Remove non-digit characters from a value and return an integer.
    Handles None, NaN, and non-string values by converting to string first.
    Returns None if no digits are found.
    """
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = re.sub(r"[^\d]", "", value)
    return int(cleaned) if cleaned else None
  
# Calculate 'calculated CFR' from Numerator and Denominator columns in the standard extraction.
def calculate_cfr(numerator_str, denominator_str):
    num = extract_int(numerator_str)
    den = extract_int(denominator_str)
    if num is not None and den is not None and den != 0:
        # Calculate as a percentage or a fraction? Here we assume fraction.
        return num / den
    else:
        return ""
  
# ------------------------ Main Extraction Loop ------------------------ #

raw_output_data = []       # For the first extraction (raw response with CFR extraction)
standard_output_data = []  # For the second extraction (standard format extraction)

for idx, row in papers_df.iterrows():
    paper_id = row["PDF"]
    true_cfr = row["True CFR"]

    paper_folder = os.path.join(papers_directory, paper_id)
    text_file_path = os.path.join(paper_folder, f"{paper_id}.txt")
    csv_file_path = os.path.join(paper_folder, f"{paper_id}.csv")

    # Read text content and CSV data (if available)
    pdf_text = read_text_file(text_file_path)
    csv_data = read_csv_as_string(csv_file_path) if os.path.exists(csv_file_path) else ""

    if not pdf_text:
        print(f"No text content for paper {paper_id}. Skipping.")
        continue

    # ---- First Extraction: Raw response ---- #
    raw_prompt = f"""
{extraction_prompt}
Table Data:
{csv_data[:10000]}
Document Text:
{pdf_text[:25000]}
"""
    print(f"\nProcessing raw extraction for paper {paper_id}...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": raw_prompt}],
            temperature=0
        )
        raw_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing paper {paper_id} (raw extraction): {e}")
        raw_response = "Error processing paper"

    overall_hosp_cfr = extract_overall_hosp_cfr(raw_response)

    raw_output_data.append({
        "Papers": paper_id,
        "True CFR": true_cfr,
        "Extracted Response": raw_response,
        "overall CFR": overall_hosp_cfr
    })

    # ---- Second Extraction: Standard format ---- #
    standard_prompt = f"""
PDF: {paper_id}
{standard_extraction_prompt}
Table Data:
{csv_data[:10000]}
Document Text:
{pdf_text[:25000]}
"""
    print(f"\nProcessing std extraction for paper {paper_id}...")
    try:
        response_std = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": standard_prompt}],
            temperature=0
        )
        std_response = response_std.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing paper {paper_id} (raw extraction): {e}")
        std_response = "Error processing paper"
    
    # Parse the text response into a dictionary.
    parsed_data = parse_standard_text(std_response)
    # Ensure that the PDF key is present.
    if "PDF" not in parsed_data or not parsed_data["PDF"]:
        parsed_data["PDF"] = paper_id
    
    standard_output_data.append(parsed_data)

# ------------------------ Save Output to Excel ------------------------ #

df_raw = pd.DataFrame(raw_output_data, columns=["Papers", "True CFR", "Extracted Response", "overall CFR"])

# Define the expected columns
standard_columns = [
    "PDF",
    "cases confirmed",
    "cases suspected",
    "# symptomatic cases",
    "# hospitalized",
    "# deaths",
    "Sample size - number of observations",
    "Sample size - number of studies",
    "Age_min",
    "Age_max",
    "Parameter Value",
    "Parameter range - lower value",
    "Parameter range - upper value",
    "Statistical approach",
    "Numerator",
    "Denominator"
]

# Create a DataFrame and add any missing columns as empty strings.
df_standard = pd.DataFrame(standard_output_data)

for col in standard_columns:
    if col not in df_standard.columns:
        df_standard[col] = ""

# Calculate the new 'calculated CFR' column
df_standard["calculated CFR"] = df_standard.apply(
    lambda row: calculate_cfr(row.get("Numerator", ""), row.get("Denominator", "")),
    axis=1
)

df_standard = df_standard[standard_columns + ["calculated CFR"]]

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_raw.to_excel(writer, sheet_name="raw response", index=False)
    df_standard.to_excel(writer, sheet_name="standard format", index=False)

print(f"\nCompleted extractions. Excel file saved to {excel_path}")