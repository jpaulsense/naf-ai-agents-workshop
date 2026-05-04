"""
YouTube → Structured Data Extractor

Lab 1 tool: Fetches a YouTube video transcript and uses Claude to extract
structured data (topics, arguments, facts, terminology, takeaways).

Usage:
    python scripts/youtube_extract.py videos.txt

Where videos.txt contains one YouTube URL per line (max 5 videos).

Output is saved to lab1_output/ as individual JSON files and a combined
markdown summary.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ---------------------------------------------------------------------------
# YouTube transcript fetching
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str | None:
    """Pull the video ID out of any common YouTube URL format."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> str | None:
    """Fetch the transcript for a YouTube video. Returns plain text or None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try English first, then any available language
        try:
            transcript = transcript_list.find_transcript(["en"])
        except Exception:
            transcript = transcript_list.find_generated_transcript(["en"])

        segments = transcript.fetch()
        # Combine all segments into plain text
        full_text = " ".join(seg.text for seg in segments)
        return full_text

    except Exception as e:
        print(f"  ERROR fetching transcript for {video_id}: {e}")
        return None


def get_video_title(video_id: str) -> str:
    """Best-effort title fetch. Falls back to the video ID."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # The API doesn't directly give titles, so fall back
        return video_id
    except Exception:
        return video_id


# ---------------------------------------------------------------------------
# Claude structured extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a structured data extraction engine. Given a
YouTube video transcript, extract the following fields. Be thorough and
specific — pull real facts, real numbers, real terminology from the content.

Return your output as valid JSON matching this exact schema:

{{
  "title": "<best guess at the video title from context>",
  "video_id": "<provided>",
  "url": "<provided>",
  "duration_estimate": "<estimate based on transcript length>",
  "key_topics": [
    "topic 1",
    "topic 2"
  ],
  "main_arguments": [
    {{
      "claim": "the main point or argument",
      "supporting_detail": "evidence or explanation given"
    }}
  ],
  "key_facts_and_statistics": [
    "specific fact, number, or statistic cited in the video"
  ],
  "terminology": [
    {{
      "term": "word or phrase",
      "definition": "how it is defined or used in the video"
    }}
  ],
  "takeaways": [
    "key takeaway 1",
    "key takeaway 2"
  ],
  "summary": "3-5 sentence summary of the entire video"
}}

Rules:
- Only include information actually stated in the transcript
- Be specific — use exact terms, names, and numbers from the video
- key_topics: 5-10 topics
- main_arguments: 3-7 arguments/claims
- key_facts_and_statistics: 5-15 specific facts or numbers
- terminology: 5-15 terms with definitions
- takeaways: 3-7 takeaways
- If the transcript is unclear or seems auto-generated with errors, do your
  best and note any uncertainty in the summary
"""


def extract_structured_data(transcript: str, video_id: str, url: str) -> dict:
    """Send transcript to Claude and get structured data back."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    # Truncate very long transcripts to stay within context limits
    max_chars = 80_000  # ~20k tokens, well within Haiku's context
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"

    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(
            content=(
                f"Video URL: {url}\n"
                f"Video ID: {video_id}\n\n"
                f"TRANSCRIPT:\n{transcript}"
            )
        ),
    ]

    response = llm.invoke(messages)

    # Parse the JSON from the response
    try:
        # Try to find JSON in the response (Claude sometimes wraps in markdown)
        text = response.content
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "Could not parse JSON from response", "raw": text}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": response.content}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def structured_data_to_markdown(data: dict) -> str:
    """Convert structured JSON data to a readable markdown document."""
    lines = []

    title = data.get("title", data.get("video_id", "Unknown"))
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Video:** {data.get('url', 'N/A')}")
    lines.append(f"**Duration:** {data.get('duration_estimate', 'Unknown')}")
    lines.append("")

    # Summary
    if data.get("summary"):
        lines.append("## Summary")
        lines.append(data["summary"])
        lines.append("")

    # Key Topics
    if data.get("key_topics"):
        lines.append("## Key Topics")
        for topic in data["key_topics"]:
            lines.append(f"- {topic}")
        lines.append("")

    # Main Arguments
    if data.get("main_arguments"):
        lines.append("## Main Arguments")
        for i, arg in enumerate(data["main_arguments"], 1):
            if isinstance(arg, dict):
                lines.append(f"**{i}. {arg.get('claim', 'N/A')}**")
                lines.append(f"   {arg.get('supporting_detail', '')}")
            else:
                lines.append(f"**{i}.** {arg}")
        lines.append("")

    # Key Facts & Statistics
    if data.get("key_facts_and_statistics"):
        lines.append("## Key Facts & Statistics")
        for fact in data["key_facts_and_statistics"]:
            lines.append(f"- {fact}")
        lines.append("")

    # Terminology
    if data.get("terminology"):
        lines.append("## Terminology")
        for entry in data["terminology"]:
            if isinstance(entry, dict):
                lines.append(f"- **{entry.get('term', 'N/A')}**: {entry.get('definition', '')}")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    # Takeaways
    if data.get("takeaways"):
        lines.append("## Takeaways")
        for i, takeaway in enumerate(data["takeaways"], 1):
            lines.append(f"{i}. {takeaway}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_VIDEOS = 5


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/youtube_extract.py <videos.txt>")
        print("")
        print("  videos.txt should contain one YouTube URL per line (max 5)")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Check your .env file.")
        sys.exit(1)

    # Read URLs
    urls = [
        line.strip()
        for line in input_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(urls) > MAX_VIDEOS:
        print(f"WARNING: Limiting to {MAX_VIDEOS} videos (found {len(urls)})")
        urls = urls[:MAX_VIDEOS]

    if not urls:
        print("No URLs found in the input file.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"YouTube → Structured Data Extractor")
    print(f"Processing {len(urls)} video(s)")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path("lab1_output")
    output_dir.mkdir(exist_ok=True)

    all_data = []
    all_markdown = []

    for i, url in enumerate(urls, 1):
        video_id = extract_video_id(url)
        if not video_id:
            print(f"[{i}/{len(urls)}] SKIP: Could not parse video ID from: {url}")
            continue

        print(f"[{i}/{len(urls)}] Processing: {url}")
        print(f"  Video ID: {video_id}")

        # Step 1: Fetch transcript
        print(f"  Fetching transcript...")
        transcript = fetch_transcript(video_id)
        if not transcript:
            print(f"  SKIP: No transcript available")
            continue

        word_count = len(transcript.split())
        print(f"  Transcript: {word_count:,} words")

        # Step 2: Extract structured data via Claude
        print(f"  Extracting structured data via Claude...")
        data = extract_structured_data(transcript, video_id, url)

        if "error" in data:
            print(f"  WARNING: {data['error']}")
        else:
            print(f"  Extracted: {len(data.get('key_topics', []))} topics, "
                  f"{len(data.get('main_arguments', []))} arguments, "
                  f"{len(data.get('key_facts_and_statistics', []))} facts, "
                  f"{len(data.get('terminology', []))} terms")

        # Save individual JSON
        json_path = output_dir / f"video_{i}_{video_id}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"  Saved: {json_path}")

        # Convert to markdown
        md = structured_data_to_markdown(data)
        all_markdown.append(md)
        all_data.append(data)

        # Small delay between API calls to be respectful
        if i < len(urls):
            time.sleep(1)

        print()

    # Save combined markdown
    if all_markdown:
        combined_path = output_dir / "all_videos_structured.md"
        separator = f"\n\n{'='*60}\n\n"
        combined_path.write_text(separator.join(all_markdown))
        print(f"Combined markdown saved: {combined_path}")

    # Save combined JSON
    if all_data:
        combined_json_path = output_dir / "all_videos_structured.json"
        combined_json_path.write_text(json.dumps(all_data, indent=2))
        print(f"Combined JSON saved: {combined_json_path}")

    print(f"\n{'='*60}")
    print(f"Done! {len(all_data)} video(s) processed")
    print(f"Output in: {output_dir}/")
    print(f"{'='*60}\n")

    # Print the structured data for the first video so they can see it
    if all_data:
        print("STRUCTURED DATA PREVIEW (first video):")
        print("-" * 40)
        print(json.dumps(all_data[0], indent=2)[:3000])
        if len(json.dumps(all_data[0], indent=2)) > 3000:
            print("  ... [truncated for display]")


if __name__ == "__main__":
    main()
