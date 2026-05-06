"""
Fetch unstructured content from multiple source types.

Supports:
  - Wikipedia articles (by topic name)
  - RSS/news feeds (by feed URL)
  - PDF documents (by URL, extracts text)
  - Local files (plain text, markdown, or PDF)

Usage:
    python scripts/fetch_source.py wikipedia "Artificial Intelligence"
    python scripts/fetch_source.py rss "https://feeds.bbci.co.uk/news/world/rss.xml"
    python scripts/fetch_source.py pdf "https://example.gov/document.pdf"
    python scripts/fetch_source.py file "transcripts/01-aircAruvnKk.md"

Output is saved to lab1_output/raw/ as plain text files.
"""

import sys
import json
import os
from pathlib import Path

import requests

OUTPUT_DIR = Path("lab1_output/raw")


def fetch_wikipedia(topic: str) -> dict:
    """Fetch a Wikipedia article's full extract by topic name."""
    # First get the summary to confirm the page exists
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    headers = {"User-Agent": "NAF-Workshop/1.0 (AI training lab)"}

    resp = requests.get(summary_url, headers=headers, timeout=15)
    if resp.status_code != 200:
        print(f"  ERROR: Wikipedia page not found for '{topic}' (status {resp.status_code})")
        print(f"  TIP: Try the exact page title. Check https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}")
        return None

    summary_data = resp.json()
    title = summary_data.get("title", topic)

    # Fetch full article text via the TextExtracts API
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    query_resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers=headers,
        timeout=15,
    )
    query_data = query_resp.json()
    pages = query_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    full_text = page.get("extract", summary_data.get("extract", ""))

    return {
        "source_type": "wikipedia",
        "title": title,
        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "text": full_text,
        "char_count": len(full_text),
    }


def fetch_rss(feed_url: str, max_articles: int = 10) -> dict:
    """Fetch articles from an RSS feed."""
    import feedparser

    feed = feedparser.parse(feed_url)

    if not feed.entries:
        print(f"  ERROR: No entries found in feed: {feed_url}")
        print(f"  TIP: Make sure the URL points to a valid RSS/Atom feed")
        return None

    feed_title = feed.feed.get("title", "Unknown Feed")
    articles = []

    for entry in feed.entries[:max_articles]:
        # Get the best available content
        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].get("value", "")
        elif hasattr(entry, "summary"):
            content = entry.summary
        elif hasattr(entry, "description"):
            content = entry.description

        # Strip HTML tags (basic)
        import re
        content = re.sub(r"<[^>]+>", "", content)

        articles.append({
            "title": entry.get("title", "Untitled"),
            "published": entry.get("published", ""),
            "content": content,
            "link": entry.get("link", ""),
        })

    # Combine into a single text block
    combined_text = f"# {feed_title}\n\n"
    for i, article in enumerate(articles, 1):
        combined_text += f"## Article {i}: {article['title']}\n"
        if article["published"]:
            combined_text += f"Published: {article['published']}\n"
        combined_text += f"\n{article['content']}\n\n---\n\n"

    return {
        "source_type": "rss",
        "title": feed_title,
        "url": feed_url,
        "text": combined_text,
        "article_count": len(articles),
        "char_count": len(combined_text),
    }


def fetch_pdf(url: str) -> dict:
    """Fetch a PDF from a URL and extract its text."""
    import pymupdf

    print(f"  Downloading PDF...")
    resp = requests.get(url, timeout=60, headers={"User-Agent": "NAF-Workshop/1.0"})
    if resp.status_code != 200:
        print(f"  ERROR: Failed to download PDF (status {resp.status_code})")
        return None

    size_mb = len(resp.content) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")

    doc = pymupdf.open(stream=resp.content, filetype="pdf")
    print(f"  Pages: {doc.page_count}")

    # Extract text from all pages
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # Get title from metadata or first line
    metadata = doc.metadata
    title = metadata.get("title", "") or full_text.split("\n")[0][:100]

    return {
        "source_type": "pdf",
        "title": title,
        "url": url,
        "text": full_text,
        "page_count": doc.page_count,
        "char_count": len(full_text),
    }


def fetch_file(filepath: str) -> dict:
    """Read a local file (text, markdown, or PDF)."""
    path = Path(filepath)

    if not path.exists():
        print(f"  ERROR: File not found: {filepath}")
        return None

    if path.suffix.lower() == ".pdf":
        import pymupdf
        doc = pymupdf.open(str(path))
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        title = doc.metadata.get("title", "") or path.stem
        return {
            "source_type": "file",
            "title": title,
            "url": str(path),
            "text": full_text,
            "page_count": doc.page_count,
            "char_count": len(full_text),
        }
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Try to extract title from first heading
        title = path.stem
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        return {
            "source_type": "file",
            "title": title,
            "url": str(path),
            "text": text,
            "char_count": len(text),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

USAGE = """
Usage: python scripts/fetch_source.py <source_type> <target>

Source types:
  wikipedia  <topic>       Fetch a Wikipedia article
  rss        <feed_url>    Fetch articles from an RSS feed
  pdf        <url>         Download and extract text from a PDF
  file       <path>        Read a local text or PDF file

Examples:
  python scripts/fetch_source.py wikipedia "Incident Command System"
  python scripts/fetch_source.py rss "https://feeds.bbci.co.uk/news/world/rss.xml"
  python scripts/fetch_source.py pdf "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"
  python scripts/fetch_source.py file "transcripts/01-aircAruvnKk.md"
"""


def main():
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    source_type = sys.argv[1].lower()
    target = " ".join(sys.argv[2:])  # Allow spaces in topic names

    print(f"\n{'='*60}")
    print(f"  Fetching: {source_type} → {target}")
    print(f"{'='*60}\n")

    # Dispatch to the right fetcher
    if source_type == "wikipedia":
        result = fetch_wikipedia(target)
    elif source_type == "rss":
        result = fetch_rss(target)
    elif source_type == "pdf":
        result = fetch_pdf(target)
    elif source_type == "file":
        result = fetch_file(target)
    else:
        print(f"ERROR: Unknown source type '{source_type}'")
        print(USAGE)
        sys.exit(1)

    if result is None:
        print("\nFetch failed. See error above.")
        sys.exit(1)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a safe filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in result["title"][:50])
    text_file = OUTPUT_DIR / f"{safe_name}.txt"
    meta_file = OUTPUT_DIR / f"{safe_name}.meta.json"

    # Write the raw text
    text_file.write_text(result["text"], encoding="utf-8")

    # Write metadata (everything except the full text)
    meta = {k: v for k, v in result.items() if k != "text"}
    meta["output_file"] = str(text_file)
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Source:  {result['title']}")
    print(f"  Type:    {result['source_type']}")
    print(f"  Size:    {result['char_count']:,} characters")
    if "page_count" in result:
        print(f"  Pages:   {result['page_count']}")
    if "article_count" in result:
        print(f"  Articles: {result['article_count']}")
    print(f"  Output:  {text_file}")
    print(f"  Meta:    {meta_file}")
    print()


if __name__ == "__main__":
    main()
