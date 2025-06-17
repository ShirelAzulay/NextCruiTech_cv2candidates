# src/loader.py

import os
import logging

# Libraries for reading PDF and DOCX
import pdfplumber
import docx

logger = logging.getLogger(__name__)

def load_all_texts(folder_path: str) -> dict:
    """
    Load all resumes or job descriptions from a folder.
    Supports .txt, .pdf and .docx. Skips other file types.
    Returns a dict: { filename: text }.
    """
    texts = {}
    for fn in os.listdir(folder_path):
        path = os.path.join(folder_path, fn)
        lower = fn.lower()

        try:
            if lower.endswith(".txt"):
                logger.debug(f"Reading TXT {fn}")
                with open(path, encoding="utf-8") as f:
                    texts[fn] = f.read()

            elif lower.endswith(".pdf"):
                logger.debug(f"Reading PDF {fn}")
                text = ""
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                texts[fn] = text

            elif lower.endswith(".docx"):
                logger.debug(f"Reading DOCX {fn}")
                doc = docx.Document(path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                texts[fn] = "\n".join(full_text)

            else:
                logger.info(f"Skipping unsupported file type: {fn}")

        except Exception as e:
            logger.warning(f"Failed to load {fn}: {e}")

    logger.info(f"{len(texts)} files loaded from {folder_path}")
    return texts
