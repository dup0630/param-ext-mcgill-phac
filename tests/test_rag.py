from LLM_interaction.rag import ChromaRetriever
from text_extractor.docint import TextExtractor

parameters = ["Total number of obese patients.",
              "Total number of non-obese participants",
              "Number of obese participants who experienced complications.",
              "Number of non-obese participants who experienced complications."]

file_path = "ALDAWOOD.pdf"
extractor = TextExtractor()
extractor.extract_text(file_path)
sections = extractor.section_chunks()

retriever = ChromaRetriever()
retriever.create_db()
retriever.add_paper_data(sections=sections, paper_id=1)
output = retriever.retrieve_from_paper(query=parameters, paper_id=1, n_results=2)

for i, q in enumerate(parameters):
    print(f"Parameter: {q}")
    print(f"Retrieved support: {output['documents'][i][0]}")