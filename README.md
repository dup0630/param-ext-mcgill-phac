# Epidemiological Parameter Extraction with LLMs

## Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Description of Components](#description-of-components)
    - [Miscellaneous Tools](#miscellaneous-tools)
    - [Automated Prompt Refining](#automated-prompt-refinement)
    - [Two-stage and RAG Extraction](#two-stage-and-rag-extraction)
    - [Formatting and Evaluation](#formatting-and-evaluation)
5. [Notes and Other Details](#notes-and-other-details)

## Project Overview

This repository contains a framework for extracting epidemiological parameters from medical research publications using large language models, with and without retrieval-augmented generation.

We have implemented various approaches to the generation of prompts, processing of papers, and interaction with ChatGPT.

This project contains three main high-level tools, described below

### Automated Prompt Generation and Refinement

We include a module to iteratively improve prompts and evaluate their effectiveness using LLM feedback and performance metrics. The algorithm starts by sending base prompt to ChatGPT and asking it for improvement suggestions. The result is then tested for extraction and its performance is evaluated. This process can be iterated multiple times, and performance metrics can be tracked across iterations.

### Two-stage and RAG Approach for Extraction

For the parameter extraction task, the **two-stage** procedure consists in sending two queries to ChatGPT. The first query contains the article text and asks the LLM to find the parameters, providing an explanation on what were the values found, where were they found, and why it thinks the values are appropriate. A second query is then sent, containing the first response, where the LLm is asked to provide a structured output with the extracted values.

We have also implemented a pipeline that combines tha latter approach with **RAG**. Instead of sending the full article text in the first query, relevant sections are retrieved from a vector database via a semantic search algorithm. The semantic search is performed using the given parameter definitions, and an argument is passed for the maximum number of sections retrieved.

### Output Formatting and Computation of Performance Metrics

The repository also contains scripts for consistent formatting of the LLM's output and computation of performance metrics based on a confusion matrix. We adopt the following performance classification:

- **"True Positive"**: The parameter appears in the paper, and the extracted value is correct.
- **"True Negative"**: The parameter does not appear in the paper, and the LLM correctly returnes `Not found`.
- **"False Positive"**: The parameter does not appear in the paper, but the LLM extracted a value.
- **"False Negative"**: The parameter appears in the paper , but the LLM returned `Not found`; **OR** the parameter appears in the paper, but the extracted value is incorrect.

The target formatting is a `.csv` file based on a standard template provided by the team at **PHAC**, which includes parameter values, ranges, and other relevant details.

## Setup Instructions

The following preparations are required for the functionality of the main tools.

1. **Dependencies**: The necessary libraries can be found in `requirements.txt` and should be installed beforehand. 
    ```bash
    pip install -r requirements.txt
    ```
2. **API credentials:** Keys and endpoints for Azure Document Intelligence and OpenAI should be set as environment variables. Alternatively, these may be stored in a `.env` file. An example template:

    ```python
    OPENAI_KEY = "key"
    OPENAI_ENDPOINT = "endpoint"
    OPENAI_VERSION = "date"
    DOCINT_KEY = "key"
    DOCINT_ENDPOINT = "endpoint"
    ```
3. **Input PDFs:** A directory containing all the PDF articles to be processed must be provided.
4. **Prompts and Parameters:** Prompts with instructions and parameter descriptions can be provided through the corresponding `.json` files in `config/`. The current template for prompts looks like this:
    ```json
    {
        "sys_prompt": "[Intro to task and general instructions]",
        "rag_sys_prompt": "[Intro to task and general instructions (RAG version)]",
        "refine_prompt": "[Instructions for formatting previous response.]"
    }   
    ```

    ```json
    {
        "parameters": [
            "Parameter name: [Name]. Description: [Description]"
            "Parameter name: [Name]. Description: [Description]"
        ]
    }
    ```
4. **True parameters:** For prompt refinement and performance evaluation, a `.csv` file containing the true parameter values for each of the treated papers. An example:
    ```
    PDF, TrueCFR, TrueLOS
    75, NA, 5
    88, NA, NA,
    7471, 0, NA,
    2146, 0, 9
    1797, 0.05, 6
    640, 0.23, NA
    6300, 4.88, NA
    ```


## Description of Components

### Miscellaneous Tools

#### `LLM_interaction/`

**Purpose**: Contains tools for interacting with large language models from OpenAI.

- `gpt_client.py`:
    - Provides a wrapper function, `ask_GPT`, for interacting with Azure-hosted OpenAI chat models.  
    - Credentials and deployment settings are read from environment variables.  
    - Accepts chat-formatted prompts and returns model responses as plain text.
 
- `rag.py`  
    - Contains the `ChromaRetriever` class, which wraps ChromaDB to enable retrieval-augmented generation (RAG).  
    - It supports database creation, document chunk insertion (e.g., PDF sections), and similarity-based retrieval.  
    - Uses Azure OpenAI embeddings to vectorize text for querying.


---

#### `text_extractor/`
**Purpose**: Handles extraction of structured and unstructured text from PDFs.

- `docint.py`  
    - Defines the `TextExtractor` class, which uses Azure Document Intelligence to parse PDF documents.  
    - Extracts line-by-line text, identifies tables, and provides access to paragraphs and section-level content.  
    - Can export combined results to plain text files for downstream processing.


---

#### `utils/`
**Purpose**: Miscellaneous utility functions.
- `load_config`: Loads JSON configuration files (e.g., prompts, parameters).
- `cleanup_dir`: Recursively removes a directory and its contents. Useful for resetting Chroma vector databases or temporary output.
- `evaluate_confusion_matrix.py`: Contains various tools for evaluating the prompts obtained by `prompt_refiner.py` (see below). For a given iteration of the pipeline, the script
    - Computes performance metrics (sensitivity, specificity, accuracy, precision, F1, MCC).
    - Displays the confusion matrix.
    - Compares against the previous iteration.
    - Flags specific papers were performance improved or got worse (e.g., from `Fail` to `Success`)
---

### Automated Prompt Refinement
#### `prompt_refiner.py`
This module provides a standalone pipeline for refining prompts used to extract epidemiological parameters from selected papers. It uses the effectiveness of different prompt formulations before generating a new one that takes into account previous fails and successes. The module tracks iteration performance over time using manual evaluation input. This can be used for any parameter given that the appropriate input files are provided.

**Usage:**
```bash
python prompt_refiner.py --folder ./pdfs
```

**Input:** 
- Path to folder containing PDF files.
- A list of parameter names as they appear in the validation data must be placed in `refiner_parameters.json`

**Workflow:**

1. Retrieve prior prompts used for a given parameter if they exists from a previous output.
2. Ask GPT to generate a refined version of the prompts.
3. Apply the refined prompt across multiple labelled documents.
4. Compute confusion matrix (using `evaluate_confusion_matrix.py`).

**Output**: All results are written to `promp_output.csv`, which has the following structure:
|Prompt |Model Name |Parameter Name |Paper Number |Extracted Parameter |True Parameter |Success/Fail |Confusion Label|Iteration |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|[prompt]|4o-mini|CFR|1538|20.5|20.5|Success| TP | 5|

This output table may accumulate across runs and supports iterative refinement and scoring.

---
### Two-stage and RAG Extraction

#### `two_stage_pipeline.py`

This pipeline applies the two-stage approach to parameter extraction on a folder of PDF files, without using RAG. It's mainly intended for testing on small papers, since it doesn't count tokens or generates chunks.
The script contains the `ParameterExtractor` class, which implements the core logic on individual papers. It's suitable for batch processing and command-line use.

**Usage:**

```bash
python two_stage_pipeline.py --folder ./pdfs --output results.csv --explanations --verbose
```

**Input:**

| Argument       | Description                                         | Default     |
|----------------|-----------------------------------------------------|-------------|
| --folder       | (Required) Path to folder containing PDF files      | N/A         |
| --output       | Path to output CSV file or directory                | output.csv  |
| --explanations | Store GPT explanations for each document            | False       |
| --verbose      | Print status messages during processing             | False       |


- The prompts and parameters to be used must be placed in the corresponding `.json` file in `config/`.

**Workflow:**
1. For each PDF in the given directory, 
    - Extract text from the PDF (using `TextExtractor`, wrapped in `ParameterExtractor`),
    - Perform first query on ChatGPT (with prompt and parameters from `config` and the full article text),
    - Refine and format the results via a second query to ChatGPT,
    - Store results.
2. Combine al results into a data frame object.
3. Export all results to a CSV file.

**Output:** 
- A `.csv` file (`two_stage_results.csv`) containing the extracted parameters for each PDF in the specified directory. 
- If `--explanations` is enabled, a text file `explanations.txt` is also saved, containing the raw GPT outputs from the first query.


---

#### `rag_pipeline.py`

This pipeline implements a retrieval-augmented generation (RAG) for batch parameter extraction on a folder of PDF files. It uses the same two-stage logic as above, but it's more efficient as it only sends retrieved relevant sections from the papers. The vector databases are managed using **ChromaDB**, and the embeddings are performed with OpenAI's *text-embedding-3-large*. The script is suitable for command-line use.

**Usage:**

```bash
python rag_pipeline.py --folder ./pdfs --output rag_results.csv --rag_n 7 --explanations --verbose
```

**Input:**

| Argument       | Description                                         | Default     |
|----------------|-----------------------------------------------------|-------------|
| --folder       | (Required) Path to folder containing PDF files      | N/A         |
| --output       | Path to output CSV file or directory                | output.csv  |
| --rag_n        | Number of most relevant sections to retrieve        | 5           |
| --explanations | Store GPT explanations for each document            | False       |
| --verbose      | Print status messages during processing             | False       |


- The prompts and parameters to be used must be placed in the corresponding `.json` file in `config/`.

**Workflow:**

1. Extracts and segments text content all PDFs in the given directory (using `TextExtractor`'s `section_chunks()` method).
2. Embeds these sections and stores them in a Chroma vector database.
1. For each PDF in the given directory, 
    - Retrieve the paper's most relevan sections from the vector database (`rag_n` sections retrieved),
    - Perform first query on ChatGPT (with prompt and parameters from `config` and the sections),
    - Refine and format the results via a second query to ChatGPT,
    - Store results.
2. Combine al results into a data frame object.
3. Export all results to a CSV file.

**Output:** 
- A `.csv` file (`rag_results.csv`) containing the extracted parameters for each PDF in the specified directory. 
- If `--explanations` is enabled, a text file `explanations.txt` is also saved, containing the raw GPT outputs from the first query.


**Notes:**
- Queries to the vector database are made using the parameter names and descriptions in `parameters.json`.
- `rag_n` sections are retrieved **per parameter**.\

---
### Formatting and Evaluation

#### `extractForAll.py`

This script implements a CFR-specific extraction workflow. It performs a two-step GPT-based extraction for hospitalized Case Fatality Rate (CFR) in measles-related studies:

- Raw Extraction: Captures unstructured LLM output using a detailed domain-specific prompt.
- Standardized Output: Parses the same paper into structured fields (e.g., sample size, age range, numerator, denominator).
- Excel Output: Results are saved in an Excel file with two sheets (raw response and standard format), along with an optional calculated CFR field.

The script can be run in two modes:
- sampled: Uses a reference CSV with true CFR values for selected papers.
- all: Runs on all available papers in the cfr_validation/paper_texts directory.

**Usage:**

```bash
python legacy_cfr_extraction.py --mode sampled
python legacy_cfr_extraction.py --mode all
```

**Input**: 
  - Processed research papers (text and CSV formats).

**Output**: Excel file with two sheets:
  - **Sheet 1**: 
    - `paperID`
    - Ground-truth hospitalized CFR (`true`)
    - Raw GPT response
    - Extracted hospitalized CFR
  - **Sheet 2**:
    - Standardized extraction format
    - Important keywords
    - Calculated CFR (`Numerator / Denominator`)
