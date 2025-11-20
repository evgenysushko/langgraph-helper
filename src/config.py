"""Configuration management and validation"""

import os
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

GEMINI_MODEL = "gemini-2.5-flash"
MAX_WEB_RESULTS = 3

class Mode(Enum):
    """Operating modes: OFFLINE (local docs) or ONLINE (fetch live docs)."""
    OFFLINE = "offline"
    ONLINE = "online"

class RetrievalMethod(Enum):
    """Document retrieval strategies: MAP (llms.txt) or MCP (MCP server)."""
    MAP = "map"
    MCP = "mcp"

class Config:
    """Application configuration with validation for environment variables and file paths."""
    def __init__(self):
        load_dotenv()

        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.docs_dir = self.data_dir / "docs"
        self.llms_txt_path = self.data_dir / "llms.txt"

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.web_search_api_key = os.getenv("WEB_SEARCH_API_KEY")

        self.mode: Mode = Mode.OFFLINE
        self.retrieval_method: RetrievalMethod = RetrievalMethod.MAP
        self.web_search_enabled: bool = False

    def validate(self) -> None:
        """Validates configuration constraints and exits with helpful error messages if invalid."""
        DOWNLOAD_CMD = "Run 'python download_docs.py' to download documentation."
        errors = []

        if self.retrieval_method == RetrievalMethod.MCP and self.mode == Mode.OFFLINE:
            errors.append(
                "MCP retrieval requires online mode.\n"
                "  Use: --mode online --retrieval mcp"
            )

        if not self.gemini_api_key:
            errors.append(
                "Missing GEMINI_API_KEY environment variable.\n"
                "  Get your free API key from: https://aistudio.google.com/app/apikey\n"
                "  Add it to your .env file or set as environment variable."
            )

        if self.retrieval_method == RetrievalMethod.MAP:
            if not self.llms_txt_path.exists():
                errors.append(
                    f"Documentation not found: {self.llms_txt_path}\n"
                    f"  {DOWNLOAD_CMD}"
                )

            if not self.docs_dir.exists() or not any(self.docs_dir.iterdir()):
                errors.append(
                    f"Documentation directory is empty: {self.docs_dir}\n"
                    f"  {DOWNLOAD_CMD}"
                )

        if self.web_search_enabled and not self.web_search_api_key:
            errors.append(
                "Missing WEB_SEARCH_API_KEY when --web-search flag is used.\n"
                "  Either:\n"
                "    - Add WEB_SEARCH_API_KEY to your .env file, or\n"
                "    - Remove --web-search flag"
            )

        if errors:
            import sys
            print("Configuration Error:", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            for error in errors:
                print(f"\n{error}", file=sys.stderr)
            print("\n" + "=" * 60, file=sys.stderr)
            raise SystemExit(1)
