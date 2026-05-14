import fitz
from PIL import Image
import pytesseract
import io


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    extracted_text = ""

    for page_number in range(len(document)):
        page = document.load_page(page_number)

        text = page.get_text()

        if text.strip():
            extracted_text += text + "\n"

        else:
            pix = page.get_pixmap()

            image = Image.open(io.BytesIO(pix.tobytes()))

            ocr_text = pytesseract.image_to_string(image)

            extracted_text += ocr_text + "\n"

    return extracted_text


def extract_text_from_image(image_path):
    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text