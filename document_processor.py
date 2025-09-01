# document_processor.py
# End-to-end: load parsed JSON -> chunk -> embed with Mistral -> save to MemorySystem

import os
import re
import json
import glob
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

import httpx
from memory_system import MemorySystem

# ---------------------------
# Setup
# ---------------------------

LOG = logging.getLogger("document_processor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MISTRAL_API_URL = os.environ.get("MISTRAL_API_URL", "https://api.mistral.ai/v1/embeddings")
MISTRAL_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

DEFAULT_BATCH_SIZE = 64
DEFAULT_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 200

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"\'])")

# ---------------------------
# Utilities
# ---------------------------

def load_json_sections(path: Path) -> List[Dict[str, Any]]:
    """Load and normalize JSON sections from file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a top-level JSON array.")
    norm = []
    for i, sec in enumerate(data):
        norm.append({
            "id": sec.get("id") or f"sec_{i:04d}",
            "anchor": sec.get("anchor"),
            "code": sec.get("code"),
            "title": sec.get("title"),
            "text": (sec.get("text") or "").strip(),
            "refs": sec.get("refs") or []
        })
    return norm


def chunk_text(text: str, title: Optional[str] = None,
               max_chars: int = DEFAULT_CHUNK_CHARS,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with enhanced cleaning."""
    if not text:
        return []

    # Clean text before processing
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    if title:
        title = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', title)
        text = title.strip() + " " + text.strip()

    # Ensure text is not empty after cleaning
    if not text.strip():
        return []

    sentences = [s.strip() for s in re.split(_SENTENCE_SPLIT_RE, text) if s.strip()]
    chunks, cur, cur_len = [], [], 0

    for sent in sentences:
        s_len = len(sent) + 1
        if cur_len + s_len <= max_chars or not cur:
            cur.append(sent)
            cur_len += s_len
        else:
            chunk_str = " ".join(cur).strip()
            if chunk_str:  # Only add non-empty chunks
                chunks.append(chunk_str)
            tail = chunk_str[-overlap:] if overlap > 0 else ""
            cur = [tail, sent] if tail else [sent]
            cur_len = len(" ".join(cur))

    if cur:
        final_chunk = " ".join(cur).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks



def build_chunk_payload(section: Dict[str, Any], chunk_text_str: str) -> Dict[str, Any]:
    title = (section.get("title") or "").strip()
    code = (section.get("code") or "").strip()
    display_title = " – ".join([p for p in [code, title] if p]) or title or code or ""
    return {
        "section_id": section.get("id"),
        "anchor": section.get("anchor"),
        "code": code,
        "title": title,
        "display_title": display_title,
        "text": chunk_text_str.strip(),
        "refs": section.get("refs") or [],
    }


def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def mistral_embed_batch(inputs: list, client, model: str = MISTRAL_MODEL) -> list:
    """Call Mistral embedding API with enhanced input validation."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

    # Enhanced cleaning and validation
    clean_inputs = []
    for s in inputs:
        if not s or not s.strip():
            continue

        # Remove problematic characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)
        cleaned = cleaned.strip()

        # Ensure reasonable length (Mistral has token limits)
        if len(cleaned) > 8000:
            cleaned = cleaned[:8000]

        # Skip very short inputs that might cause issues
        if len(cleaned) < 10:
            continue

        clean_inputs.append(cleaned)

    if not clean_inputs:
        return []

    try:
        payload = {"model": model, "input": clean_inputs}
        resp = client.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return embeddings

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            LOG.warning("422 error for batch of %d inputs. Response: %s",
                       len(clean_inputs), e.response.text)

            # Handle 422 by processing inputs individually
            if len(clean_inputs) > 1:
                LOG.info("Splitting 422 batch into individual requests")
                embeddings = []
                for single_input in clean_inputs:
                    try:
                        single_embedding = mistral_embed_batch([single_input], client, model)
                        embeddings.extend(single_embedding)
                    except Exception as single_error:
                        LOG.error("Failed to embed single input: %s", single_error)
                        # Return zero vector or skip this input
                        embeddings.append([0.0] * 1024)  # Adjust dimension as needed
                return embeddings
            else:
                LOG.error("Single input causing 422 error: %s", clean_inputs[:100])
                return [[0.0] * 1024]  # Return zero vector for problematic input

        elif e.response.status_code == 400 and len(clean_inputs) > 1:
            LOG.warning("400 Bad Request: splitting batch of %d into halves", len(clean_inputs))
            mid = len(clean_inputs) // 2
            return (
                mistral_embed_batch(clean_inputs[:mid], client, model)
                + mistral_embed_batch(clean_inputs[mid:], client, model)
            )
        raise



def save_chunk_to_memory(memory: MemorySystem, doc_id: str, chunk: Dict[str, Any], embedding: list):
    """Save a chunk with embedding as JSON in SQLite."""
    try:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        embedding_json = json.dumps(embedding)

        if hasattr(memory, "save_document"):
            memory.save_document(
                doc_id=doc_id,
                section_id=chunk.get("section_id"),
                title=chunk.get("title"),
                text=chunk.get("text"),
                refs=chunk.get("refs", []),
                embedding=embedding_json,
            )
            LOG.debug("Saved chunk %s/%s", doc_id, chunk.get("section_id"))
            return
        raise RuntimeError("MemorySystem does not expose save_document")
    except Exception as e:
        LOG.error("Save failed for chunk %s: %s", chunk.get('section_id'), e)
        raise


def process_json_file(
    json_path: Path,
    memory: MemorySystem,
    doc_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Load JSON sections, chunk them, embed them, and save into memory system."""

    LOG.info("Processing %s -> doc_id=%s", json_path, doc_id)

    # 1️⃣ Load sections
    sections = load_json_sections(json_path)
    LOG.info("Loaded %d sections from %s", len(sections), json_path)

    # 2️⃣ Chunk each section
    all_chunks = []
    for section in sections:
        text = section.get("text", "")
        if not text:
            continue

        title = section.get("title")
        chunks = chunk_text(text, title=title, max_chars=chunk_chars, overlap=chunk_overlap)
        for chunk_text_str in chunks:
            chunk_payload = build_chunk_payload(section, chunk_text_str)
            all_chunks.append(chunk_payload)

    LOG.info("Created %d chunks from %d sections", len(all_chunks), len(sections))

    if dry_run:
        LOG.info("DRY RUN: Would process %d chunks", len(all_chunks))
        return len(sections), len(all_chunks)

    # 3️⃣ Generate embeddings and save in batches
    saved_count = 0
    with httpx.Client() as client:
        for i, batch_chunks in enumerate(batched(all_chunks, batch_size)):
            texts = [chunk["text"] for chunk in batch_chunks]

            try:
                LOG.debug("Processing batch %d/%d (%d chunks)",
                          i + 1, (len(all_chunks) + batch_size - 1) // batch_size, len(batch_chunks))

                embeddings = mistral_embed_batch(texts, client)  # ✅ fixed function

                for chunk, embedding in zip(batch_chunks, embeddings):
                    save_chunk_to_memory(memory, doc_id, chunk, embedding)  # ✅ forces float list
                    saved_count += 1

                LOG.info("Processed batch %d: %d/%d chunks saved",
                         i + 1, saved_count, len(all_chunks))
                time.sleep(0.1)

            except Exception as e:
                LOG.error("Failed to process batch %d (chunks %d-%d): %s",
                          i + 1, saved_count, saved_count + len(batch_chunks), e)
                raise

    LOG.info("Successfully processed %s: %d sections -> %d chunks",
             json_path.name, len(sections), saved_count)

    return len(sections), saved_count



def discover_json_inputs(path_like: str) -> List[Path]:
    p = Path(path_like)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(Path(x) for x in glob.glob(str(p / '**' / '*.json'), recursive=True))
    raise FileNotFoundError(f"Input path not found: {path_like}")


def main():
    parser = argparse.ArgumentParser(description="COGS Document Processor")
    parser.add_argument("--input", required=True)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--db-path", default=str(Path("data") / "cogs_memory.db"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not MISTRAL_API_KEY and not args.dry_run:
        raise RuntimeError("Set MISTRAL_API_KEY or use --dry-run")

    inputs = discover_json_inputs(args.input)
    LOG.info("Found %d JSON files", len(inputs))

    memory = MemorySystem(db_path=args.db_path)

    total_sections = total_chunks = 0
    for path in inputs:
        doc_id = args.doc_id or f"{path.parent.name}-{path.stem}".lower()
        sec_count, chunk_count = process_json_file(path, memory, doc_id,
                                                   batch_size=args.batch_size,
                                                   chunk_chars=args.chunk_chars,
                                                   chunk_overlap=args.chunk_overlap,
                                                   dry_run=args.dry_run)
        total_sections += sec_count
        total_chunks += chunk_count

    LOG.info("COMPLETE: Files=%d, sections=%d, chunks_saved=%d", len(inputs), total_sections, total_chunks)


if __name__ == "__main__":
    main()
