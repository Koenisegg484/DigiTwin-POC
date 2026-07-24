import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from git import Repo
from langchain_core.documents import Document

from app.loaders.base_loader import BaseLoader
from app.loaders.code_loader import CodeLoader
from app.loaders.markdown_loader import MarkdownLoader
from app.loaders.pdf_loader import PDFLoader
from app.loaders.text_loader import TextLoader


# Directories that should never be walked into.
SKIP_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "myenv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".idea",
    ".vscode",
    ".chroma",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".nuxt",
    "coverage",
    ".gradle",
    ".terraform",
}

# File extensions that must never be loaded.
SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".webp",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2",
    ".lock", ".pyc", ".pyo", ".class", ".o", ".obj", ".wasm",
}

# Files by basename that must never be loaded.
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

# Extensions routed to each loader.
_MARKDOWN_EXTS = {".md", ".markdown", ".mdx", ".rst"}
_TEXT_EXTS = {".txt", ".text", ".log", ".csv", ".tsv"}
_PDF_EXTS = {".pdf"}
_CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
    ".go", ".rs", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx",
    ".cs", ".rb", ".php", ".swift", ".scala", ".m", ".sh", ".bash",
    ".zsh", ".ps1", ".sql", ".r", ".dart", ".lua", ".pl", ".clj",
    ".ex", ".exs", ".erl", ".hs", ".ml", ".fs", ".vb",
}


class GitHubLoader(BaseLoader):
    """Clones a GitHub repository and dispatches each file to the right loader.

    Follows Single Responsibility: this loader does not parse file contents.
    It clones, walks, filters, and delegates to Markdown/Text/PDF/Code loaders.
    """

    def __init__(self, url: str, clone_dir: str | None = None):
        self.url = url
        self.clone_dir = Path(clone_dir) if clone_dir else None

    def load(self) -> list[Document]:
        if self.clone_dir is not None:
            return self._load_into(self.clone_dir)

        # Clone to a temp directory and clean up afterwards.
        #TODO: remember to delete the code repo later
        tmp = Path(tempfile.mkdtemp(prefix="onboard_clone_"))
        try:
            return self._load_into(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _load_into(self, dest: Path) -> list[Document]:
        Repo.clone_from(self.url, str(dest), depth=1, single_branch=True)
        documents: list[Document] = []
        for path in self._walk(dest):
            if self._should_skip(path):
                continue
            docs = self._dispatch(path)
            if docs:
                documents.extend(docs)
        return documents

    @staticmethod
    def _walk(root: Path) -> Iterable[Path]:
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            # Skip anything inside a forbidden directory.
            if any(part in SKIP_DIRS for part in entry.relative_to(root).parts):
                continue
            yield entry

    @staticmethod
    def _should_skip(path: Path) -> bool:
        if path.name in SKIP_FILES:
            return True
        if path.suffix.lower() in SKIP_EXTS:
            return True
        return False

    @staticmethod
    def _dispatch(path: Path) -> list[Document]:
        ext = path.suffix.lower()
        str_path = str(path)

        if ext in _MARKDOWN_EXTS:
            return MarkdownLoader(str_path).load()
        if ext in _TEXT_EXTS:
            return TextLoader(str_path).load()
        if ext in _PDF_EXTS:
            return PDFLoader(str_path).load()
        if ext in _CODE_EXTS:
            return CodeLoader(str_path).load()
        # Unknown extension: ignore for now.
        return []
