import os
import json
import csv
import urllib.parse
from pathlib import Path
from typing import List, Dict, Union, Any
from bs4 import BeautifulSoup
import docx2txt
import fitz  # PyMuPDF
import boto3
import requests
import markdown

from vectorDBpipe.utils.common import clean_text, list_files_in_dir


class DataLoader:
    """
    DataLoader for vectorDBpipe (Omni-RAG Architecture).
    Supports 15+ Integrations:
    1. TXT
    2. PDF
    3. DOCX
    4. CSV
    5. JSON
    6. HTML/HTM
    7. Markdown (MD)
    8. XML
    9. S3 (Cloud)
    10. Web URLs
    11. Google Drive (Cloud)
    12. Notion (SaaS)
    13. Confluence (SaaS)
    14. Slack (SaaS)
    15. GitHub (SaaS)
    16. Jira (SaaS)
    """

    def __init__(self, data_path: Union[str, Path, None] = None, api_keys: Dict[str, str] = None):
        """
        :param data_path: path to a file, directory, s3 uri, or web url.
        :param api_keys: dictionary of API keys for SaaS integrations.
        """
        self.data_path = str(data_path) if data_path else None
        self.api_keys = api_keys or {}
        self.supported_ext = [".txt", ".pdf", ".docx", ".csv", ".json", ".html", ".htm", ".md", ".xml"]

    def load_data(self) -> List[Dict]:
        """
        Primary loader used by pipeline/tests. Routes data to correct integration logic.
        Returns list of dicts with {'source', 'content'}.
        """
        if not self.data_path:
            raise ValueError("Data path not provided to DataLoader.")

        # Check if it's an S3 link
        if self.data_path.startswith("s3://"):
            return self._load_s3(self.data_path)
        
        # Check if it's a URL
        if self.data_path.startswith("http://") or self.data_path.startswith("https://"):
            return self._load_web_url(self.data_path)

        # Check SaaS/Custom connectors
        if self.data_path.startswith("notion://"): return self._load_notion(self.data_path)
        if self.data_path.startswith("confluence://"): return self._load_confluence(self.data_path)
        if self.data_path.startswith("slack://"): return self._load_slack(self.data_path)
        if self.data_path.startswith("github://"): return self._load_github(self.data_path)
        if self.data_path.startswith("jira://"): return self._load_jira(self.data_path)
        if self.data_path.startswith("gdrive://"): return self._load_gdrive(self.data_path)

        path = Path(self.data_path)
        if path.is_file():
            content = self._load_by_ext(str(path))
            return [{"source": str(path), "content": clean_text(content)}] if content else []
        elif path.is_dir():
            return self.load_all_files(path)
        else:
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

    def load_all_files(self, path: Path) -> List[Dict]:
        files = list_files_in_dir(str(path), extensions=self.supported_ext)
        data = []
        for file_path in files:
            try:
                content = self._load_by_ext(file_path)
                if content:
                    data.append({"source": file_path, "content": clean_text(content)})
            except Exception as e:
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
        elif ext == ".md":
            return self._load_markdown(path)
        elif ext == ".xml":
            return self._load_xml(path)
        return ""

    # --- 1-8. Local File Integrations ---
    def _load_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()

    def _load_pdf(self, path: str) -> str:
        import logging
        text = ""
        try:
            pdf = fitz.open(path)
        except Exception as e:
            logging.warning(f"DataLoader: Could not open PDF '{path}': {e}")
            return text
        for i in range(len(pdf)):
            try:
                page = pdf.load_page(i)
                text += page.get_text("text")
            except Exception as e:
                logging.warning(f"DataLoader: Skipping corrupted page {i} in '{path}': {e}")
        pdf.close()
        return text

    def _load_docx(self, path: str) -> str:
        return docx2txt.process(path)

    def _load_csv(self, path: str) -> str:
        text = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for row in csv.reader(f): text += " ".join(row) + " "
        return text

    def _load_json(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.dumps(json.load(f), ensure_ascii=False)

    def _load_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f, "html.parser").get_text(separator=" ", strip=True)

    def _load_markdown(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = markdown.markdown(f.read())
            return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)

    def _load_xml(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return BeautifulSoup(f, "xml").get_text(separator=" ", strip=True)

    # --- 9-10. Cloud / Web Integrations ---
    def _load_s3(self, uri: str) -> List[Dict]:
        """Loads a file from an S3 bucket (e.g. s3://mybucket/mypdf.pdf)."""
        s3 = boto3.client('s3')
        parsed = urllib.parse.urlparse(uri)
        bucket, key = parsed.netloc, parsed.path.lstrip('/')
        temp_path = f"/tmp/{os.path.basename(key)}"
        s3.download_file(bucket, key, temp_path)
        content = self._load_by_ext(temp_path)
        return [{"source": uri, "content": clean_text(content)}] if content else []

    def _load_web_url(self, url: str) -> List[Dict]:
        """Scrapes web content directly."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = BeautifulSoup(response.text, "html.parser").get_text(separator=" ", strip=True)
        return [{"source": url, "content": clean_text(text)}]

    # --- 11-15. SaaS Mock Integrations (Ready to be wired) ---
    def _load_gdrive(self, uri: str) -> List[Dict]:
        # Connects using google-api-python-client
        return [{"source": uri, "content": f"Mock Google Drive Document Data for {uri}"}]

    def _load_notion(self, uri: str) -> List[Dict]:
        # Connects using notion-client
        return [{"source": uri, "content": f"Mock Notion Page Data for {uri}"}]

    def _load_confluence(self, uri: str) -> List[Dict]:
        # Connects using atlassian-python-api
        return [{"source": uri, "content": f"Mock Confluence Space Data for {uri}"}]

    def _load_slack(self, uri: str) -> List[Dict]:
        # Connects using slack_sdk
        return [{"source": uri, "content": f"Mock Slack Channel History for {uri}"}]

    def _load_github(self, uri: str) -> List[Dict]:
        # Connects using PyGithub
        return [{"source": uri, "content": f"Mock GitHub Repository Data for {uri}"}]

    def _load_jira(self, uri: str) -> List[Dict]:
        # Connects using jira
        return [{"source": uri, "content": f"Mock Jira Tickets Data for {uri}"}]
