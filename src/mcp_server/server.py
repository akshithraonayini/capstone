from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("Enterprise Knowledge Server")

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data/documents"


@mcp.tool
def list_knowledge_documents() -> list[str]:
    """List all available enterprise knowledge documents."""

    if not KNOWLEDGE_DIR.exists():
        return []

    return [
        file.name
        for file in KNOWLEDGE_DIR.iterdir()
        if file.is_file()
    ]


@mcp.tool
def read_knowledge_document(filename: str) -> str:
    """Read an enterprise knowledge document."""

    file_path = KNOWLEDGE_DIR / filename

    if not file_path.exists():
        return f"Document not found: {filename}"

    if not file_path.is_file():
        return f"Not a valid file: {filename}"

    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()