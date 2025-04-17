from text_extractor.docint import TextExtractor

extractor = TextExtractor(output_dir="test_output")
path = input("Enter file path: ")
extractor.extract_text(path, verbose=True)
extractor.export_text(output_name="test.txt")