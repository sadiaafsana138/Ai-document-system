from app.ocr import extract_text_from_pdf

text = extract_text_from_pdf("data/raw/sample.pdf")

print(text)