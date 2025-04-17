from docint import TextExtractor

file_path = "ALDAWOOD.pdf"
extractor = TextExtractor()
extractor.extract_text(file_path, verbose=True)
results = extractor.result_object
sections = extractor.section_chunks()
print(dir(sections[0]))