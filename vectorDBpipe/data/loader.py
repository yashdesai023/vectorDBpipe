import os
import json
import csv
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
import docx2txt
import fitz  # PyMuPDF

from vectorDBpipe.utils.common import clean_text, list_files_in_dir


class DataLoader:
    """
    Loads text data from multiple document formats.
    Supported: .txt, .pdf, .docx, .csv, .json, .html
    """

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.supported_ext = [".txt", ".pdf", ".docx", ".csv", ".json", ".html"]

    def load_all_files(self) -> List[Dict]:
        """
        Load and extract text from all supported file types.
        Returns list of dicts with {'source', 'content'}.
        """
        files = list_files_in_dir(str(self.input_dir), extensions=self.supported_ext)
        data = []
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            try:
                if ext == ".txt":
                    content = self._load_txt(file_path)
                elif ext == ".pdf":
                    content = self._load_pdf(file_path)
                elif ext == ".docx":
                    content = self._load_docx(file_path)
                elif ext == ".csv":
                    content = self._load_csv(file_path)
                elif ext == ".json":
                    content = self._load_json(file_path)
                elif ext in [".html", ".htm"]:
                    content = self._load_html(file_path)
                else:
                    continue

                if content:
                    data.append({"source": file_path, "content": clean_text(content)})

            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")
        return data

    def _load_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_pdf(self, path: str) -> str:
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text("text")
        return text

    def _load_docx(self, path: str) -> str:
        return docx2txt.process(path)

    def _load_csv(self, path: str) -> str:
        text = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                text += " ".join(row) + " "
        return text

    def _load_json(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = json.load(f)
        # Flatten nested JSONs to string
        return json.dumps(content, ensure_ascii=False)

    def _load_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator=" ", strip=True)
