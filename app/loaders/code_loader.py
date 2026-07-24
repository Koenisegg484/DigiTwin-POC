from pathlib import Path
from langchain_core.documents import Document
from app.loaders.base_loader import BaseLoader


# Maps common code extensions to a language label.
_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".m": "objc",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".sql": "sql",
    ".r": "r",
    ".dart": "dart",
    ".lua": "lua",
    ".pl": "perl",
    ".vim": "vim",
    ".el": "elisp",
    ".clj": "clojure",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".fs": "fsharp",
    ".vb": "vb",
}


class CodeLoader(BaseLoader):
    """Reads a source file as plain text into one Document.

    Language-aware chunking will be added later; for now code is treated
    as plain text to keep retrieval honest.
    """

    def __init__(self, file_path: str):
        self.path = Path(file_path)

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding="utf-8")
        language = _LANGUAGE_MAP.get(self.path.suffix.lower(), "unknown")
        doc = Document(
            page_content=text,
            metadata={
                "source": str(self.path),
                "file_type": "code",
                "language": language,
            },
        )
        return [doc]
