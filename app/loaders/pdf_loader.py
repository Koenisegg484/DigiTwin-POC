from pathlib import Path
import fitz
from langchain_core.documents import Document
from app.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    
    def __init__(self, file_path: str):
        self.path = Path(file_path)

    def load(self) -> list[Document]:
        pages_text: list[str] = []
        with fitz.open(self.path) as doc:
            for page in doc:
                pages_text.append(page.get_text())
        joined = "\n".join(pages_text).strip()
        return [
            Document(
                page_content=joined,
                metadata={
                    "source": str(self.path),
                    "file_type": "pdf",
                    "pages": len(pages_text),
                },
            )
        ]
