"""
YouTube → Structured Knowledge Base Builder

Lab 1 tool: Fetches YouTube video transcripts and uses Claude to build
a dual-format knowledge base:

  - JSON: Discrete, machine-readable facts (terminology, statistics,
    metadata). Think lookup tables, search indexes, quiz generation.

  - YAML: Contextual, narrative knowledge (arguments with reasoning,
    conceptual explanations, connections between ideas). Think
    conversation context, RAG retrieval, agent background knowledge.

This mirrors how production AI systems store knowledge — discrete data
for precision, contextual data for understanding.

Usage:
    python scripts/youtube_extract.py videos.txt

Where videos.txt contains one YouTube URL per line (max 5 videos).

Output is saved to lab1_output/:
  - Per-video JSON and YAML files
  - Combined knowledge_base.json (all discrete data)
  - Combined knowledge_base.yaml (all contextual knowledge)
  - knowledge_base_summary.md (human-readable overview)
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

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        full_text = " ".join(snippet.text for snippet in transcript)
        return full_text

    except Exception as e:
        print(f"  ERROR fetching transcript for {video_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Claude extraction — JSON (discrete facts)
# ---------------------------------------------------------------------------

JSON_EXTRACTION_PROMPT = """You are a structured data extraction engine.
Given a YouTube video transcript, extract DISCRETE, MACHINE-READABLE facts.

Think of this as building a database or lookup table. Every field should be
a specific, self-contained piece of information that could be searched,
filtered, or used to generate a quiz question.

Return valid JSON matching this exact schema:

{
  "metadata": {
    "title": "<best guess at video title from context>",
    "video_id": "<provided>",
    "url": "<provided>",
    "channel": "<best guess at channel/speaker name>",
    "duration_estimate": "<estimate based on transcript length>",
    "difficulty_level": "<beginner|intermediate|advanced>",
    "primary_domain": "<the main field or discipline>"
  },
  "key_topics": [
    "topic 1",
    "topic 2"
  ],
  "terminology": [
    {
      "term": "exact word or phrase",
      "definition": "concise definition as used in the video",
      "category": "<concept|process|tool|metric|role|other>"
    }
  ],
  "key_facts": [
    {
      "fact": "specific claim, number, or statistic",
      "context": "brief note on where/why this was mentioned"
    }
  ],
  "people_and_references": [
    {
      "name": "person, organization, or work referenced",
      "relevance": "why they were mentioned"
    }
  ],
  "takeaways": [
    "actionable or memorable takeaway"
  ]
}

Rules:
- ONLY include information explicitly stated in the transcript
- Be specific — exact terms, exact numbers, exact names
- terminology: 8-20 entries with precise, concise definitions
- key_facts: 8-20 specific facts, statistics, or claims
- people_and_references: any people, books, papers, tools, or orgs mentioned
- takeaways: 5-10 entries
- If the transcript has auto-generated errors, do your best and note in metadata"""


# ---------------------------------------------------------------------------
# Claude extraction — YAML (contextual knowledge)
# ---------------------------------------------------------------------------

YAML_EXTRACTION_PROMPT = """You are a knowledge base builder. Given a YouTube
video transcript, extract CONTEXTUAL, NARRATIVE knowledge — the kind of
understanding that lets someone have an intelligent conversation about this
topic.

Think of this as building background knowledge for an AI assistant. The output
should preserve reasoning chains, explain WHY things matter, and show how
concepts connect to each other.

Return valid YAML (not JSON) matching this structure:

---
topic: "<main topic of the video>"
source:
  title: "<best guess at video title>"
  video_id: "<provided>"
  url: "<provided>"
  channel: "<best guess>"

conceptual_overview: |
  <3-5 paragraph narrative summary of the video's content. Write this as if
  you're briefing someone who needs to understand the topic well enough to
  discuss it intelligently. Include the main thesis, the logical flow of the
  argument, and why this topic matters. Use the speaker's own framing and
  examples where possible.>

arguments:
  - claim: "<main point or argument #1>"
    reasoning: |
      <2-3 paragraph explanation of the reasoning and evidence behind this
      claim. Include the examples, analogies, or data the speaker used.
      Preserve the logical chain — what leads to what.>
    implications: |
      <What does this mean in practice? Why should someone care?>
    related_concepts:
      - "<concept that connects to this argument>"
      - "<another related concept>"

  - claim: "<main point or argument #2>"
    reasoning: |
      <same depth as above>
    implications: |
      <same>
    related_concepts:
      - "<concept>"

connections:
  - from: "<concept A>"
    to: "<concept B>"
    relationship: |
      <Explain how these concepts relate. What's the causal or logical
      link? This is the kind of knowledge that lets someone answer
      'how does X relate to Y?' questions.>

practical_applications:
  - scenario: "<real-world situation where this knowledge applies>"
    how_it_applies: |
      <How would someone use this knowledge in this scenario?>

open_questions:
  - "<question the video raises but doesn't fully answer>"
  - "<area where the speaker acknowledges uncertainty>"

speaker_perspective: |
  <Brief note on the speaker's background, biases, or point of view
  that colors their presentation. This helps an agent contextualize
  the knowledge — e.g., 'the speaker is a practitioner, not a
  researcher, so examples are practical rather than theoretical.'>

Rules:
- Write contextually rich content — paragraphs, not bullets
- Preserve the speaker's reasoning chains and examples
- arguments: 3-6 entries with DEEP reasoning sections
- connections: 3-8 entries showing how concepts link together
- practical_applications: 2-5 entries
- open_questions: 2-5 entries
- Use | for multi-line strings in YAML
- If the transcript has errors, do your best and note in conceptual_overview"""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_json_knowledge(transcript: str, video_id: str, url: str) -> dict:
    """Extract discrete, machine-readable facts as JSON."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    max_chars = 80_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"

    messages = [
        SystemMessage(content=JSON_EXTRACTION_PROMPT),
        HumanMessage(content=f"Video URL: {url}\nVideo ID: {video_id}\n\nTRANSCRIPT:\n{transcript}"),
    ]

    response = llm.invoke(messages)

    try:
        text = response.content
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse JSON from response", "raw": text}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": response.content}


def extract_yaml_knowledge(transcript: str, video_id: str, url: str) -> str:
    """Extract contextual, narrative knowledge as YAML."""
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

    max_chars = 80_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED]"

    messages = [
        SystemMessage(content=YAML_EXTRACTION_PROMPT),
        HumanMessage(content=f"Video URL: {url}\nVideo ID: {video_id}\n\nTRANSCRIPT:\n{transcript}"),
    ]

    response = llm.invoke(messages)

    # Clean up: remove markdown code fences if Claude wraps the YAML
    text = response.content.strip()
    if text.startswith("```yaml"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ---------------------------------------------------------------------------
# Summary markdown generation
# ---------------------------------------------------------------------------

def generate_summary_markdown(json_data: dict, yaml_text: str, index: int) -> str:
    """Generate a human-readable markdown summary showing both formats."""
    lines = []
    meta = json_data.get("metadata", {})

    title = meta.get("title", json_data.get("title", f"Video {index}"))
    lines.append(f"# Video {index}: {title}")
    lines.append("")
    lines.append(f"**URL:** {meta.get('url', 'N/A')}")
    lines.append(f"**Channel:** {meta.get('channel', 'Unknown')}")
    lines.append(f"**Duration:** {meta.get('duration_estimate', 'Unknown')}")
    lines.append(f"**Difficulty:** {meta.get('difficulty_level', 'Unknown')}")
    lines.append(f"**Domain:** {meta.get('primary_domain', 'Unknown')}")
    lines.append("")

    # JSON section — show what discrete data looks like
    lines.append("---")
    lines.append("")
    lines.append("## Discrete Knowledge (JSON)")
    lines.append("")
    lines.append("*This is the machine-readable data — the kind you'd use for")
    lines.append("search indexes, quiz generation, glossaries, and structured queries.*")
    lines.append("")

    if json_data.get("key_topics"):
        lines.append("### Topics")
        for topic in json_data["key_topics"]:
            lines.append(f"- {topic}")
        lines.append("")

    if json_data.get("terminology"):
        lines.append("### Terminology")
        for entry in json_data["terminology"]:
            if isinstance(entry, dict):
                cat = f" `{entry.get('category', '')}`" if entry.get("category") else ""
                lines.append(f"- **{entry.get('term', 'N/A')}**{cat}: {entry.get('definition', '')}")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    if json_data.get("key_facts"):
        lines.append("### Key Facts & Statistics")
        for entry in json_data["key_facts"]:
            if isinstance(entry, dict):
                lines.append(f"- {entry.get('fact', 'N/A')}")
                if entry.get("context"):
                    lines.append(f"  *Context: {entry['context']}*")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    if json_data.get("people_and_references"):
        lines.append("### People & References")
        for entry in json_data["people_and_references"]:
            if isinstance(entry, dict):
                lines.append(f"- **{entry.get('name', 'N/A')}**: {entry.get('relevance', '')}")
            else:
                lines.append(f"- {entry}")
        lines.append("")

    if json_data.get("takeaways"):
        lines.append("### Takeaways")
        for i, t in enumerate(json_data["takeaways"], 1):
            lines.append(f"{i}. {t}")
        lines.append("")

    # YAML section — show what contextual knowledge looks like
    lines.append("---")
    lines.append("")
    lines.append("## Contextual Knowledge (YAML)")
    lines.append("")
    lines.append("*This is the narrative knowledge — the kind you'd feed to an AI")
    lines.append("agent so it can have an intelligent conversation about the topic,")
    lines.append("understand reasoning chains, and explain how concepts connect.*")
    lines.append("")
    lines.append("```yaml")
    # Show first ~3000 chars of YAML to keep the summary readable
    preview = yaml_text[:3000]
    if len(yaml_text) > 3000:
        preview += "\n# ... [see full YAML file for complete contextual knowledge]"
    lines.append(preview)
    lines.append("```")
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
        print("")
        print("Output:")
        print("  lab1_output/")
        print("    video_1_<id>.json       — discrete facts (machine-readable)")
        print("    video_1_<id>.yaml       — contextual knowledge (narrative)")
        print("    knowledge_base.json     — combined discrete data")
        print("    knowledge_base.yaml     — combined contextual knowledge")
        print("    knowledge_base_summary.md — human-readable overview")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

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
    print(f"  YouTube → Structured Knowledge Base Builder")
    print(f"  Processing {len(urls)} video(s)")
    print(f"{'='*60}")
    print()
    print("Two output formats per video:")
    print("  JSON = Discrete facts (terminology, stats, metadata)")
    print("         → for lookups, quizzes, search indexes")
    print("  YAML = Contextual knowledge (arguments, reasoning, connections)")
    print("         → for conversations, RAG, agent context")
    print()

    # Create output directory
    output_dir = Path("lab1_output")
    output_dir.mkdir(exist_ok=True)

    all_json_data = []
    all_yaml_texts = []
    all_summaries = []

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
            print(f"  SKIP: No transcript available for this video")
            print(f"  (Video may not have captions enabled)")
            continue

        word_count = len(transcript.split())
        est_minutes = word_count // 150  # rough speaking rate
        print(f"  Transcript: {word_count:,} words (~{est_minutes} min of speech)")

        # Step 2: Extract JSON (discrete facts)
        print(f"  Extracting discrete facts (JSON)...")
        json_data = extract_json_knowledge(transcript, video_id, url)

        if "error" in json_data:
            print(f"  WARNING (JSON): {json_data['error']}")
        else:
            terms = len(json_data.get("terminology", []))
            facts = len(json_data.get("key_facts", []))
            topics = len(json_data.get("key_topics", []))
            print(f"  JSON: {topics} topics, {terms} terms, {facts} facts")

        # Save individual JSON
        json_path = output_dir / f"video_{i}_{video_id}.json"
        json_path.write_text(json.dumps(json_data, indent=2))

        # Step 3: Extract YAML (contextual knowledge)
        print(f"  Extracting contextual knowledge (YAML)...")
        yaml_text = extract_yaml_knowledge(transcript, video_id, url)

        # Count YAML sections as a rough quality check
        arg_count = yaml_text.count("- claim:")
        conn_count = yaml_text.count("- from:")
        print(f"  YAML: {arg_count} arguments, {conn_count} connections")

        # Save individual YAML
        yaml_path = output_dir / f"video_{i}_{video_id}.yaml"
        yaml_path.write_text(yaml_text)

        print(f"  Saved: {json_path}")
        print(f"  Saved: {yaml_path}")

        # Generate summary markdown
        summary_md = generate_summary_markdown(json_data, yaml_text, i)
        all_summaries.append(summary_md)

        all_json_data.append(json_data)
        all_yaml_texts.append(yaml_text)

        # Delay between videos to avoid rate limits
        if i < len(urls):
            print(f"  Pausing briefly before next video...")
            time.sleep(2)

        print()

    # -----------------------------------------------------------------------
    # Save combined outputs
    # -----------------------------------------------------------------------

    if all_json_data:
        combined_json_path = output_dir / "knowledge_base.json"
        combined_json_path.write_text(json.dumps(all_json_data, indent=2))
        print(f"Combined JSON knowledge base: {combined_json_path}")

    if all_yaml_texts:
        combined_yaml_path = output_dir / "knowledge_base.yaml"
        separator = "\n\n---\n# " + "="*56 + "\n\n"
        combined_yaml_path.write_text(separator.join(all_yaml_texts))
        print(f"Combined YAML knowledge base: {combined_yaml_path}")

    if all_summaries:
        summary_path = output_dir / "knowledge_base_summary.md"
        page_break = f"\n\n{'='*60}\n\n"
        summary_path.write_text(page_break.join(all_summaries))
        print(f"Human-readable summary: {summary_path}")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------

    total_terms = sum(len(d.get("terminology", [])) for d in all_json_data)
    total_facts = sum(len(d.get("key_facts", [])) for d in all_json_data)
    total_args = sum(y.count("- claim:") for y in all_yaml_texts)
    total_conns = sum(y.count("- from:") for y in all_yaml_texts)

    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE BASE COMPLETE")
    print(f"{'='*60}")
    print(f"  Videos processed:    {len(all_json_data)}")
    print(f"  Output directory:    {output_dir}/")
    print(f"")
    print(f"  DISCRETE DATA (JSON):")
    print(f"    Terminology entries:  {total_terms}")
    print(f"    Key facts:           {total_facts}")
    print(f"")
    print(f"  CONTEXTUAL DATA (YAML):")
    print(f"    Arguments with reasoning: {total_args}")
    print(f"    Concept connections:      {total_conns}")
    print(f"{'='*60}")
    print()
    print("Files in lab1_output/:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  {f.name:40s} {size_str}")
    print()
    print("NEXT STEPS:")
    print("  1. Open the .json files to see discrete, searchable facts")
    print("  2. Open the .yaml files to see contextual, narrative knowledge")
    print("  3. Open knowledge_base_summary.md for a side-by-side overview")
    print("  4. Use this structured data to build a curriculum, quiz, or briefing")
    print()


if __name__ == "__main__":
    main()
