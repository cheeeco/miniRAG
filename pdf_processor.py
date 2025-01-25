import os

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def extract_text_from_pdf(path_to_pdf: str):
    path_to_txt = path_to_pdf.replace(".pdf", ".txt")
    if os.path.exists(path_to_txt):
        return path_to_txt
    
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(path_to_pdf)
    text, _, images = text_from_rendered(rendered)
    with open(path_to_txt, "w") as fp:
        fp.writelines(text)

    return path_to_txt