from pathlib import Path
from langchain_core.documents import Document
from app.loaders.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):

    def __init__(self, file_path: str):
        self.path = Path(file_path)

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={
                "source": str(self.path),
                "file_type": "markdown",
            },
        )
        return [doc]
