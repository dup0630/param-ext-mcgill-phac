"""
docint.py

Provides the `TextExtractor` class for extracting structured and unstructured content from PDFs
using Azure Document Intelligence (prebuilt-layout model).

This module supports:
- Line-by-line text extraction
- Table detection and serialization
- Section and paragraph segmentation
- Exporting extracted content to disk

Environment Variables Required:
- DOCINT_KEY: Azure Document Intelligence API key
- DOCINT_ENDPOINT: Endpoint URL for the Document Intelligence resource

Dependencies:
- azure-ai-documentintelligence
- pandas
- python-dotenv
"""

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
import base64
import pandas as pd

class TextExtractor:
    def __init__(self, output_dir="DocIntOutput"):
        self.output_dir = output_dir
        self.extraction_performed = False
        self.text = ""
        self.tables = []
        self.full_text = ""

        # Initialize Azure client
        load_dotenv()
        self.key = os.getenv("DOCINT_KEY")
        self.endpoint = os.getenv("DOCINT_ENDPOINT")
        if not self.key or not self.endpoint:
            raise ValueError("OPENAI_KEY and/or OPENAI_ENDPOINT not set in environment variables.")
        self.client = DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))

    def extract_text(self, pdf_path: str, verbose: bool = False) -> None:
        """
        Extracts text and tables from a PDF file using Azure Document Intelligence and stores the results.
        This method:
            - Reads the PDF located at `pdf_path`
            - Extracts raw text line-by-line and stores it in `self.text`
            - Extracts tables, converts them to JSON strings, and stores them in `self.tables`
            - Combines text and table data into `self.full_text`
            - Sets `self.extraction_performed` to True after completion
        Requires Azure Document Intelligence API key and endpoint to be set in environment variables as DOCINT_KEY and DOCINT_ENDPOINT.
        """
        # Open and process pdf
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode('utf-8')
        poller = self.client.begin_analyze_document("prebuilt-layout", {"base64Source": encoded_pdf})
        result = poller.result()
        self.result_object = result

        # Extract text from the analysis result
        extracted_text = ""
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"
        self.text = extracted_text

        for table in result.tables:
            table_data = {}
            # Extract table into a dictionary format
            for cell in table.cells:
                if cell.row_index not in table_data:
                    table_data[cell.row_index] = {}
                table_data[cell.row_index][cell.column_index] = cell.content
            
            df = pd.DataFrame.from_dict(table_data, orient="index").sort_index(axis=1) # Maybe this intermediate step isn't necessary?
            json_str = df.to_json(orient="records")
            self.tables.append(json_str)
        
        joint_tables = "\n\n\n".join(self.tables)
        self.full_text = self.text + "\n\n\nTables:\n" + joint_tables
        self.extraction_performed = True

        if verbose:
            print("Text successfully extracted")

    def export_text(self, output_name: str = "extracted_text.txt") -> None:
        """
        Exports the extracted text to a .txt file. If an extraction has not been performed, raises a ValueError.
        """
        if not self.extraction_performed: raise ValueError('An extraction has not yet been performed.')

        os.mkdir(self.output_dir)
        text_path = os.path.join(self.output_dir, output_name)
        if os.path.exists(text_path):
            print(f"Warning: {text_path} already exists. Overwriting.")
            os.remove(text_path)
        # Export the text
        with open(text_path, "w") as text_file:
            text_file.write(self.full_text)
    
    def paragraph_chunks(self) -> list[str]:
        """
        Splits the extracted text into paragraphs.
        """
        paragraphs: list[str] = []
        for paragraph in self.result_object.paragraphs:
            paragraphs.append(paragraph.content)
        return paragraphs
    
    def section_chunks(self) -> list[str]:
        """
        Splits the extracted text into sections.
        """
        sections: list[str] = []
        for section in self.result_object.sections:
            # The attribute 'section.elements' contains the paragraphs in the section.
            # Each paragraph is indexed as '/paragraphs/0', '/paragraphs/1', etc.
            # We need to extract the paragraph index from the string and use it to get the content.
            elements: list[str] = section.elements
            contents: str = ""
            for paragraph in elements:
                paragraph_index = int(paragraph.split("/")[-1])
                paragraph_content = self.result_object.paragraphs[paragraph_index].content
                contents += paragraph_content + "\n"
            sections.append(contents)
        return sections

if __name__ == "__main__":
    # Example usage
    path = input("Enter file path: ")
    extractor = TextExtractor()
    extractor.extract_text(path, verbose=True)
    text = extractor.full_text
    print(text)
    extractor.export_text()

    # cleanup_dir("DocIntOutput")
