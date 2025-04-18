# Epidemiological Parameter Extraction with LLMs

This repository contains a framework for extracting epidemiological parameters from medical research publications using large language models, with and without retrieval-augmented generation.

This work was developed in collaboration with the **Public Health Agency of Canada** to support automated knowledge extraction for public health surveillance and evidence synthesis.

## Contents
1. [Project Structur and Main Tools](#project-structure-and-main-tools)
2. [Setup Instructions](#setup-instructions)
3. [Usage and Examples](#usage)
4. [Description of Components](#description-of-components)
5. [Notes and Other Details]()

## Project Structure and Main Tools

This project is organized into modular components. These modules are combined into **two** main pipelines, described in the **Usage** section below.


## Description of Components
### `LLM_interaction/`
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

### `text_extractor/`
**Purpose**: Handles extraction of structured and unstructured text from PDFs.

- `docint.py`  
    - Defines the `TextExtractor` class, which uses Azure Document Intelligence to parse PDF documents.  
    - Extracts line-by-line text, identifies tables, and provides access to paragraphs and section-level content.  
    - Can export combined results to plain text files for downstream processing.


---

### `config/`
**Purpose**: Configuration files for prompts and parameter definitions.

- `parameters.json`:  
   Defines the parameters to extract from research articles.  
- `prompts.json`:  
   Contains LLM prompts to be used for the parameret extraction task.

---

### `.env`
Sensitive credentials for APIs such as Azure Document Intelligence and OpenAI should be stored here. An example template:

```python
OPENAI_KEY = "key"
OPENAI_ENDPOINT = "endpoint"
OPENAI_VERSION = "date"
DOCINT_KEY = "key"
DOCINT_ENDPOINT = "endpoint"
```
---

### `utils/`
**Purpose**: Miscellaneous utility functions.
- `load_config`: Loads JSON configuration files (e.g., prompts, parameters).
- `cleanup_dir`: Recursively removes a directory and its contents. Useful for resetting Chroma vector databases or temporary output.
---

### Top-Level Pipelines

- `two_stage_pipeline.py`
    - Contains the `ParameterExtractor` class, which implements the core logic of the two_stage pipeline (see **Usage**).  
    - Runs the full two_stage parameter extraction pipeline across all PDFs in a specified directory (see **Usage**).  
    - Uses prompts and parameter definitions from the `config/` folder and outputs the structured results as a CSV.  
    - Outputs can be exported to text files for inspection or evaluation.
    - Suitable for batch processing and command-line use.

- `rag_pipeline.py`  
    - Implements a retrieval-augmented generation (RAG) pipeline for batch parameter extraction from medical PDFs.  
    - Uses vector similarity search to retrieve relevant sections for each parameter, then performs a two-stage GPT query to extract and refine information.  
    - Outputs a structured CSV containing results for all documents in a directory.
    - Suitable for batch processing and command-line use.


These scripts are designed to process multiple PDF files within a directory using the respective pipeline approach.


## Setup Instructions

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure environment**:  
   Create a `.env` file with your OpenAI and Azure credentials.

3. **Prepare inputs**:  
   Place your research PDFs into a designated directory (e.g., `./papers/`).



## Usage

### Option A: Two-stage Extraction

This approach consists in processesing a folder of PDF files, and applying a two-stage LLM pipeline to extract epidemiological parameters from each document.

Workflow:

1. For each PDF in the given directory, 
    - Extract text from the PDF,
    - Perform initial parameter extraction using an LLM,
    - Refine and format the results via a second LLM query,
    - Store results.
2. Combine al results into a data frame object.
3. Export all results to a CSV file.

It leverages the `ParameterExtractor` class from `double_layer_core` and uses configuration files located in the `config/` directory for parameter definitions and prompt templates.

CLI usage:

```bash
python double_layer_pipeline.py --folder ./pdfs --output results.csv --verbose
```

### Option B: RAG Implementation

This approach consists in using RAG to process the contents of a folder of PDF files and extract specified parameters using GPT. It combines section-based text extraction (from `TextExtractor` in `docint.py`), vector embedding and retrieval via ChromaDB and OpenAI's text-embedding-3-large, and a two-step LLM prompting strategy (initial extraction + refinement, as in Option A).

Workflow:
1. Extract text from each PDF and split it into sections.
2. Embed and store each section in a Chroma vector database.
3. For each paper, retrieve the most relevant sections for each target parameter.
4. Use GPT to generate and refine parameter extractions based on the retrieved context.
5. Export all results to a CSV file.

CLI usage:
```bash
python rag_pipeline.py --folder ./pdfs --output rag_results.csv --rag_n 7 --verbose
```
You may customize parameter definitions and prompts by editing files in the config/ directory.

## Notes
The LLMs used in this project (for text embedding and chat completions) are accessed via OpenAI’s API.

For retrieval, the project currently uses a vector store (ChromaDB) and a basic retriever. This can be extended to other architectures.