"""
rag_pipeline.py

This script processes a folder of PDF files using a retrieval-augmented generation (RAG) pipeline to extract specified epidemiological parameters using GPT. It combines vector-based section retrieval via ChromaDB with a two-step LLM prompting strategy (initial extraction + refinement).

Workflow:
1. Extract text from each PDF and split it into sections.
2. Embed and store each section in a Chroma vector database.
3. For each paper, retrieve the most relevant sections for each target parameter.
4. Use GPT to generate and refine parameter extractions based on the retrieved context.
5. Export all results to a CSV file.

Dependencies:
- Azure Document Intelligence
- Azure OpenAI
- ChromaDB
- pandas
- tiktoken
- json, os
"""

import argparse
from utils.utils import load_config, cleanup_dir
from LLM_interaction.rag import ChromaRetriever
from LLM_interaction.gpt_client import ask_GPT
from text_extractor.docint import TextExtractor
import os
import json
import pandas as pd
import tiktoken

def main(folder_path: str, verbose: bool = False, output_path: str = "output.csv", rag_n: int = 5) -> None:
    """
    Processes all PDF files in a folder, creates vector database for RAG, extracts specified parameters using GPT, 
    and saves the results to a CSV file.
    """
    prompts: dict = load_config("config/prompts.json")
    sys_prompt: str = prompts["rag_sys_prompt"]
    refine_prompt: str = prompts["refine_prompt"]
    parameters: list[str] = load_config("config/parameters.json")["parameters"]

    text_extractor = TextExtractor()
    retriever = ChromaRetriever()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    retriever.create_db()

    # Add papers to vector database
    n: int = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            text_extractor.extract_text(file_path)
            sections = text_extractor.section_chunks()
            for i in list(range(len(sections))):
                retriever.add_paper_data(sections=[sections[i]], paper_id=filename, section_ids=[i])
                if verbose:
                    tokens = len(tokenizer.encode(",".join(sections[i])))
                    print(f"Embedding {filename}, section {i}: {tokens} tokens.")
            n+=1
            if verbose:
                print(f"File {n} ({filename}) embedded.")

    # GPT queries with RAG
    data: list[dict] = []
    titles: list[str] = []
    n = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.lower().endswith('.pdf'):
            # Perform vector search for section retrieval
            rag_output = retriever.retrieve_from_paper(parameters, filename, rag_n)
            rag_context = ["\n".join(rag_output["documents"][i]) for i in range(0, len(parameters))]

            # First pass for explanations
            first_prompt = [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"These are the requested parameters:\n{parameters}\n\n"},
                            {"role": "user", "content": f"These are the relevant extracts: \n{rag_context}"}]
            first_response = ask_GPT(prompt=first_prompt)

            # Second pass for formatting
            second_prompt = [{"role": "system", "content": refine_prompt},
                             {"role": "user", "content": f"These are the requested parameters:\n{parameters}\n\n"},
                             {"role": "user", "content": f"This is the text:\n{first_response}"}]
            refined_response = ask_GPT(prompt=second_prompt)

            found_parameters: dict = json.loads(refined_response)
            # Add results to list
            data.append(found_parameters)
            # Label with file name
            titles.append(filename)
            n+=1
            if verbose:
                print(f"File {n} processed.")
        else:
            next
    if verbose:
        print(f"{n} files processed.")

    df =  pd.DataFrame(data)
    df["Paper"] = titles
    df.to_csv(output_path, index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline on a folder of PDFs.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing PDF files.")
    parser.add_argument("--output", default="output.csv", help="Path to save the output CSV file.")
    parser.add_argument("--rag_n", type=int, default=5, help="Number of sections to retrieve per parameter.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    main(folder_path=args.folder, output_path=args.output, rag_n=args.rag_n, verbose=args.verbose)
