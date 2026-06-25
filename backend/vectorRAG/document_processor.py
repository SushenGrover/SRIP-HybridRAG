"""
Document Processor
===================
Extracts text from financial PDFs and splits into overlapping chunks
with metadata, following the HybridRAG paper methodology (Section 2.1).

Pipeline:
  PDF  →  per-page text extraction (PyMuPDF)
       →  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
       →  metadata attachment (source file, page number, chunk index)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text with its metadata."""
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}


class DocumentProcessor:
    """Process financial PDFs into retrievable chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # PDF text extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text_ocr(doc: fitz.Document) -> List[dict]:
        """Fallback OCR extraction for image-based PDFs."""
        try:
            import pytesseract
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                "OCR dependencies not available. Install 'pytesseract' and 'pillow', "
                "and ensure Tesseract is installed on the system."
            ) from exc

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            if text.strip():
                pages.append({"page_number": page_num + 1, "text": text})
        return pages

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[dict]:
        """
        Extract text page-by-page from a PDF.
        Returns list of {page_number, text}.
        """
        doc = fitz.open(pdf_path)
        try:
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pages.append({"page_number": page_num + 1, "text": text})

            if pages:
                logger.info("Extracted %d non-empty pages from %s", len(pages), os.path.basename(pdf_path))
                return pages

            # OCR fallback for scanned/image-only PDFs
            logger.warning("No text extracted from %s. Attempting OCR fallback.", os.path.basename(pdf_path))
            pages = DocumentProcessor._extract_text_ocr(doc)
            logger.info("OCR extracted %d non-empty pages from %s", len(pages), os.path.basename(pdf_path))
            return pages
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Chunking with metadata
    # ------------------------------------------------------------------
    def process(self, pdf_path: str) -> List[Chunk]:
        """
        Full pipeline: extract text → chunk → attach metadata.
        """
        filename = os.path.basename(pdf_path)
        pages = self.extract_text_from_pdf(pdf_path)

        all_chunks: List[Chunk] = []
        global_idx = 0

        for page_info in pages:
            page_num = page_info["page_number"]
            text = page_info["text"]

            # Split this page's text into chunks
            splits = self.splitter.split_text(text)

            for local_idx, chunk_text in enumerate(splits):
                chunk = Chunk(
                    text=chunk_text.strip(),
                    metadata={
                        "source_file": filename,
                        "page_number": page_num,
                        "chunk_index": global_idx,
                        "page_chunk_index": local_idx,
                    },
                )
                all_chunks.append(chunk)
                global_idx += 1

        logger.info(
            "Processed '%s': %d pages → %d chunks",
            filename,
            len(pages),
            len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Utility: get document summary info
    # ------------------------------------------------------------------
    @staticmethod
    def get_pdf_info(pdf_path: str) -> dict:
        """Return basic metadata about a PDF."""
        doc = fitz.open(pdf_path)
        info = {
            "filename": os.path.basename(pdf_path),
            "page_count": len(doc),
            "file_size_kb": round(os.path.getsize(pdf_path) / 1024, 1),
        }
        # Try to get title from metadata
        meta = doc.metadata
        if meta and meta.get("title"):
            info["title"] = meta["title"]
        doc.close()
        return info
