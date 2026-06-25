import os

# Root directory (current folder)
ROOT_DIR = os.getcwd()
OUTPUT_FILE = os.path.join(ROOT_DIR, "codes.txt")

# Code file extensions to include
CODE_EXTENSIONS = {
    ".py", ".ipynb", ".cpp", ".c", ".h", ".hpp",
    ".java", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".xml",
    ".sql", ".sh", ".bat", ".ps1",
    ".md", ".txt", ".tex",
    ".go", ".rs", ".php", ".rb", ".swift",
    ".kt", ".kts", ".r"
}

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".idea",
    ".vscode",
    "build",
    "dist",
    ".next",
    ".cache",
    ".pytest_cache"
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for root, dirs, files in os.walk(ROOT_DIR):

        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in sorted(files):

            if file == "codes.txt":
                continue

            path = os.path.join(root, file)

            ext = os.path.splitext(file)[1].lower()

            if ext not in CODE_EXTENSIONS:
                continue

            relative_path = os.path.relpath(path, ROOT_DIR)

            out.write("=" * 100 + "\n")
            out.write(f"FILE: {relative_path}\n")
            out.write("=" * 100 + "\n\n")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except UnicodeDecodeError:
                try:
                    with open(path, "r", encoding="latin-1") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"\n[Could not read file: {e}]\n")
            except Exception as e:
                out.write(f"\n[Could not read file: {e}]\n")

            out.write("\n\n\n")

print(f"Done! All code has been saved to:\n{OUTPUT_FILE}")