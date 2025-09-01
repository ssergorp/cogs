#!/usr/bin/env python3
"""
Parse Dallas ordinance HTML into structured JSON sections.

- Detects section headers like: <div class="Section toc-destination ..."><a id="JD_7-5.15">...</a> SEC. 7-5.15. TITLE...</div>
- Gathers all following sibling content blocks until the next Section header.
- Extracts:
  - id: sequential section id (sec_0001 etc.)
  - anchor: anchor id (e.g., "JD_7-5.15")
  - code: anchor title/number if available (e.g., "7-5.15")
  - title: header line text (e.g., "SEC. 7-5.15. REQUIREMENTS FOR ...")
  - text: concatenated body text for the section
  - refs: list of internal/external references found in the header+body
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple

from bs4 import BeautifulSoup, Tag, NavigableString


NOISE_TEXTS = {"notes"}  # lines that are likely boilerplate/noise (lowercased compare)


def clean_text(s: str) -> str:
    # Normalize whitespace, strip NBSPs and extra spaces.
    s = s.replace("\xa0", " ").replace("\u00a0", " ").replace("\u200b", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def node_text(node: Tag) -> str:
    # Pull text with gentle separation between blocks.
    txt = node.get_text(separator="\n", strip=True)
    txt = clean_text(txt)
    # Filter trivial noise-only lines
    lines = [ln for ln in txt.splitlines() if ln.strip().lower() not in NOISE_TEXTS]
    return "\n".join(lines).strip()


def parse_jump_to(to_attr: str) -> Optional[str]:
    """
    Extract an internal hash from Link's 'to' prop, e.g.
    "{{ pathname: '/codes/dallas/latest/dallas_tx/0-0-0-101999', hash: '#JD_7-5.14' }}"
    -> "#JD_7-5.14"
    """
    if not to_attr:
        return None
    m = re.search(r"hash:\s*['\"]([^'\"]+)['\"]", to_attr)
    if m:
        return m.group(1)
    # Fallback: try to find #JD_... anywhere
    m = re.search(r"(#JD_[\w\.-]+)", to_attr)
    if m:
        return m.group(1)
    return None


def collect_refs(container: Tag) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []

    # Standard anchor tags
    for a in container.find_all("a"):
        href = a.get("href", "")
        text = clean_text(a.get_text(strip=True))
        if not href and not text:
            continue
        if href.startswith("#"):
            refs.append({"type": "internal", "target": href.lstrip("#"), "text": text})
        elif href:
            refs.append({"type": "external", "target": href, "text": text})

    # React-style <Link ...> components
    for link in container.find_all(lambda t: isinstance(t, Tag) and t.name.lower() == "link"):
        to_attr = link.get("to", "") or ""
        text = clean_text(link.get_text(strip=True))
        internal = parse_jump_to(to_attr)
        if internal:
            refs.append({"type": "internal", "target": internal.lstrip("#"), "text": text})
        else:
            # If we couldn't parse, still keep a vague ref
            if to_attr or text:
                refs.append({"type": "unknown", "target": to_attr, "text": text})

    return refs


def is_section_header(div: Tag) -> bool:
    if not div or div.name != "div":
        return False
    classes = set(div.get("class", []))
    # Many pages use "Section toc-destination rbox"
    return "Section" in classes and "toc-destination" in classes


def is_content_block(div: Tag) -> bool:
    if not div or div.name != "div":
        return False
    classes = set(div.get("class", []))
    # Dallas pages often use "rbox Normal-Level" for lines/paragraphs
    # but there can be other content-holding "rbox" siblings too.
    if "clearfix" in classes:
        return False
    if "Section" in classes and "toc-destination" in classes:
        return False
    return "rbox" in classes or "Normal-Level" in classes or not classes


def extract_header_info(header_div: Tag) -> Tuple[Optional[str], Optional[str], str, List[Dict[str, Any]]]:
    """
    Returns (anchor_id, code, title_text, refs_from_header)
    """
    anchor_id = None
    code = None

    a = header_div.find("a", attrs={"id": True})
    if a:
        anchor_id = a.get("id")
        code = a.get("title") or a.get("name")  # often "7-5.15"

    title_text = node_text(header_div)
    refs = collect_refs(header_div)

    return anchor_id, code, title_text, refs


def parse_html_file(html_path: Path) -> List[Dict[str, Any]]:
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    headers = soup.select("div.Section.toc-destination")
    tables = soup.select("table")
    sections: List[Dict[str, Any]] = []

    # --- 1. Process tables first ---
    for i, tbl in enumerate(tables, start=1):
        text = "\n".join(node_text(cell) for cell in tbl.find_all(["td", "th"]))
        # crude filter for zoning or CS tables
        if "CS" in text or "District" in text:
            sections.append({
                "id": f"table_{i:04d}",
                "anchor": None,
                "code": None,
                "title": "Zoning Table",
                "text": text,
                "refs": collect_refs(tbl),
            })

    # --- 2. If no headers, fallback to a single-section document ---
    if not headers:
        body = soup.body or soup
        body_text = node_text(body)
        refs = collect_refs(body)
        if body_text:
            sections.append({
                "id": "sec_0001",
                "anchor": None,
                "code": None,
                "title": "Document",
                "text": body_text,
                "refs": refs,
            })
        return sections  # Done if there are no headers

    # --- 3. Process section headers ---
    for i, header in enumerate(headers):
        anchor_id, code, title_text, header_refs = extract_header_info(header)

        # Collect subsequent sibling blocks until the next header
        body_text_parts: List[str] = []
        refs = list(header_refs)

        for sib in header.next_siblings:
            if isinstance(sib, NavigableString):
                continue
            if isinstance(sib, Tag):
                if is_section_header(sib):
                    break
                if is_content_block(sib):
                    t = node_text(sib)
                    if t:
                        body_text_parts.append(t)
                    refs.extend(collect_refs(sib))

        body_text = clean_text("\n".join(body_text_parts))

        # Avoid repeating the title as body text
        if body_text.strip().lower() == title_text.strip().lower():
            body_text = ""

        # Skip completely empty sections
        if not title_text and not body_text:
            continue

        sections.append({
            "id": f"sec_{i+1:04d}",
            "anchor": anchor_id,
            "code": code,
            "title": title_text,
            "text": body_text,
            "refs": refs,
        })

    return sections


def write_json(out_path: Path, data: Any, pretty: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def process_input(input_path: Path, outdir: Optional[Path], pretty: bool) -> None:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(list(input_path.glob("*.html")))

    if not files:
        print("⚠️ No .html files found at", input_path)
        return

    for html_file in files:
        sections = parse_html_file(html_file)
        if outdir:
            out_path = outdir / (html_file.stem + ".json")
        else:
            out_path = html_file.with_suffix(".json")

        write_json(out_path, sections, pretty=pretty)
        print(f"✅ {html_file.name}: {len(sections)} sections → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Parse ordinance HTML into JSON sections.")
    p.add_argument("input", help="Path to a single .html file or a directory containing .html files.")
    p.add_argument("--outdir", type=Path, default=None, help="Optional output directory for .json files.")
    p.add_argument("--no-pretty", action="store_true", help="Write compact JSON (no pretty-print).")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    process_input(input_path, args.outdir, pretty=not args.no_pretty)


if __name__ == "__main__":
    main()
