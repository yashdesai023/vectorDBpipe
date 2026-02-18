import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Union
from bs4 import BeautifulSoup
import docx2txt
import fitz  # PyMuPDF

from vectorDBpipe.utils.common import clean_text, list_files_in_dir


class DataLoader:
    """
    Loads text data from multiple document formats.
    Supported: .txt, .pdf, .docx, .csv, .json, .html
    Can be initialized with a single file path or a directory.
    """

    def __init__(self, data_path: Union[str, Path, None] = None):
        """
        :param data_path: path to a file or directory containing files.
        """
        self.data_path = Path(str(data_path)) if data_path else None
        self.supported_ext = [".txt", ".pdf", ".docx", ".csv", ".json", ".html", ".htm"]

    def load_data(self) -> List[Dict]:
        """
        Primary loader used by pipeline/tests.
        If `data_path` is a file -> load that single file.
        If `data_path` is a directory -> load all supported files under it.
        Returns list of dicts with {'source', 'content'}.
        """
        if not self.data_path:
            raise ValueError("Data path not provided to DataLoader.")

        if self.data_path.is_file():
            # load a single file
            content = self._load_by_ext(str(self.data_path))
            return [{"source": str(self.data_path), "content": clean_text(content)}] if content else []
        elif self.data_path.is_dir():
            return self.load_all_files()
        else:
            # Path doesn't exist
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    def load_all_files(self) -> List[Dict]:
        """
        Load and extract text from all supported file types in a directory.
        """
        files = list_files_in_dir(str(self.data_path), extensions=self.supported_ext)
        data = []
        for file_path in files:
            try:
                content = self._load_by_ext(file_path)
                if content:
                    data.append({"source": file_path, "content": clean_text(content)})
            except Exception as e:
                # keep going on error but log to console (pipeline logger will capture more later)
                print(f"[ERROR] Failed to load {file_path}: {e}")
        return data

    def _load_by_ext(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".txt":
            return self._load_txt(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        elif ext == ".csv":
            return self._load_csv(path)
        elif ext == ".json":
            return self._load_json(path)
        elif ext in [".html", ".htm"]:
            return self._load_html(path)
        else:
            return ""

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
        return json.dumps(content, ensure_ascii=False)

    def _load_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator=" ", strip=True)
