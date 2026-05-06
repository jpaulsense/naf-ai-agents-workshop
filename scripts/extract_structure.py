"""
Extract structured data from raw text using Claude AI.

Takes raw text (from fetch_source.py) and produces structured JSON
with key topics, definitions, facts, decision points, and relationships.

Usage:
    python scripts/extract_structure.py lab1_output/raw/Artificial_Intelligence.txt
    python scripts/extract_structure.py lab1_output/raw/Artificial_Intelligence.txt --mode curriculum
    python scripts/extract_structure.py lab1_output/raw/Artificial_Intelligence.txt --mode briefing
    python scripts/extract_structure.py lab1_output/raw/Artificial_Intelligence.txt --mode guide

Modes:
    curriculum  - Training curriculum with modules, study guide, quiz (default)
    briefing    - Intelligence/situation briefing format
    guide       - Interactive decision guide with branching paths

Output is saved to lab1_output/structured/ as JSON.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

OUTPUT_DIR = Path("lab1_output/structured")


# ---------------------------------------------------------------------------
# Extraction prompts for each mode
# ---------------------------------------------------------------------------

CURRICULUM_PROMPT = """You are a structured data extraction engine for training curriculum development.

Given unstructured text content, extract and organize it into a JSON structure suitable for building a training curriculum.

Return ONLY valid JSON (no markdown fencing) with this structure:
{
  "title": "Topic title",
  "summary": "2-3 sentence overview of the content",
  "modules": [
    {
      "title": "Module title",
      "learning_objective": "One sentence — what the learner will be able to do",
      "key_topics": ["topic1", "topic2", "topic3"],
      "difficulty": "foundational|intermediate|advanced"
    }
  ],
  "terminology": [
    {"term": "Term", "definition": "Clear, concise definition"}
  ],
  "key_facts": [
    {"fact": "Statement of fact", "context": "Why this matters"}
  ],
  "common_misconceptions": [
    {"misconception": "What people get wrong", "correction": "The truth"}
  ],
  "quiz_questions": [
    {
      "question": "Question text",
      "type": "multiple_choice|short_answer",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "The correct answer",
      "explanation": "Why this is correct"
    }
  ],
  "sources_used": "Brief note on what content was analyzed"
}

Rules:
- Extract ONLY information present in the source text
- Do NOT add facts, definitions, or concepts not in the source
- Sequence modules from foundational to advanced
- Generate 8-12 terminology entries
- Generate 5-8 key facts
- Generate 10 quiz questions (7 multiple choice, 3 short answer)
- Keep definitions concise (1-2 sentences max)"""

BRIEFING_PROMPT = """You are a structured data extraction engine for intelligence briefings.

Given unstructured text content (news articles, reports, documents), extract and organize it into a JSON structure suitable for a situation briefing.

Return ONLY valid JSON (no markdown fencing) with this structure:
{
  "title": "Briefing title",
  "classification": "UNCLASSIFIED",
  "date_produced": "Today's date",
  "summary": "Executive summary — 3-5 sentences covering the key situation",
  "key_events": [
    {
      "event": "What happened",
      "date": "When (if available)",
      "location": "Where (if available)",
      "significance": "Why it matters"
    }
  ],
  "actors": [
    {
      "name": "Person/org/country",
      "role": "Their role in the situation",
      "actions": "What they did or are doing"
    }
  ],
  "threats_and_risks": [
    {
      "threat": "Description of threat",
      "likelihood": "high|medium|low",
      "impact": "What could happen"
    }
  ],
  "trends": [
    {"trend": "Pattern or direction", "evidence": "Supporting data points"}
  ],
  "recommendations": [
    {"action": "Recommended action", "priority": "immediate|short_term|long_term", "rationale": "Why"}
  ],
  "information_gaps": ["What we don't know yet"],
  "sources_used": "Brief note on what content was analyzed"
}

Rules:
- Extract ONLY information present in the source text
- Do NOT speculate beyond what the sources state
- Flag uncertainty — if something is unclear, note it in information_gaps
- Maintain objectivity — present facts, not opinions
- Use clear, concise military-style language"""

GUIDE_PROMPT = """You are a structured data extraction engine for interactive decision guides.

Given unstructured text content (manuals, procedures, regulations), extract and organize it into a JSON structure suitable for building an interactive decision-tree guide.

Return ONLY valid JSON (no markdown fencing) with this structure:
{
  "title": "Guide title",
  "purpose": "What this guide helps the user accomplish",
  "audience": "Who this guide is for",
  "decision_points": [
    {
      "id": "dp_1",
      "question": "Question the user must answer to proceed",
      "context": "Brief explanation of why this matters",
      "options": [
        {
          "label": "Option A label",
          "description": "When to choose this option",
          "leads_to": "dp_2 or a result_id"
        }
      ]
    }
  ],
  "results": [
    {
      "id": "result_1",
      "title": "Result/recommendation title",
      "guidance": "The specific guidance or procedure to follow",
      "references": ["Source section/chapter references"],
      "warnings": ["Important caveats or safety notes"]
    }
  ],
  "terminology": [
    {"term": "Term", "definition": "Definition"}
  ],
  "quick_reference": [
    {"topic": "Topic", "key_point": "The essential takeaway"}
  ],
  "sources_used": "Brief note on what content was analyzed"
}

Rules:
- Extract ONLY information present in the source text
- Build decision trees that reflect actual branching logic in the source material
- Each decision point should have 2-4 options
- Results should contain actionable, specific guidance
- Include warnings for safety-critical information
- Cross-reference between related decision paths where appropriate"""


PROMPTS = {
    "curriculum": CURRICULUM_PROMPT,
    "briefing": BRIEFING_PROMPT,
    "guide": GUIDE_PROMPT,
}


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_structure(text: str, mode: str) -> dict:
    """Send text to Claude and get structured JSON back."""
    prompt = PROMPTS[mode]

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=8192)

    # Truncate if too long (Haiku context is 200k but we want fast responses)
    max_chars = 50000
    if len(text) > max_chars:
        print(f"  Text is {len(text):,} chars — truncating to {max_chars:,} for processing")
        text = text[:max_chars]

    print(f"  Sending to Claude ({mode} mode)...")
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Extract structured data from this content:\n\n{text}"),
    ])

    # Parse the JSON response
    content = response.content.strip()
    # Remove markdown code fencing if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]  # Remove first line
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        structured = json.loads(content)
        return structured
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open structures
        repaired = content
        # Count open braces/brackets
        opens = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        # Truncate to last complete entry (find last complete object)
        if repaired.rfind("},") > 0:
            repaired = repaired[:repaired.rfind("},") + 1]
        elif repaired.rfind("}") > 0:
            repaired = repaired[:repaired.rfind("}") + 1]
        # Close remaining structures
        repaired += "]" * open_brackets + "}" * opens
        try:
            structured = json.loads(repaired)
            print(f"  NOTE: Response was truncated — repaired JSON (some data may be incomplete)")
            return structured
        except json.JSONDecodeError as e:
            print(f"  WARNING: Claude returned invalid JSON that could not be repaired.")
            print(f"  TIP: Try running again — the response may have been cut off by network issues.")
            return {"raw_response": content, "parse_error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

USAGE = """
Usage: python scripts/extract_structure.py <input_file> [--mode <mode>]

Modes:
  curriculum  - Training curriculum (modules, study guide, quiz) [default]
  briefing    - Intelligence briefing (events, actors, threats, recommendations)
  guide       - Interactive decision guide (branching decision tree)

Examples:
  python scripts/extract_structure.py lab1_output/raw/Artificial_Intelligence.txt
  python scripts/extract_structure.py lab1_output/raw/BBC_World.txt --mode briefing
  python scripts/extract_structure.py lab1_output/raw/NIST_CSF.txt --mode guide
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    mode = "curriculum"

    # Parse --mode flag
    if "--mode" in sys.argv:
        mode_idx = sys.argv.index("--mode") + 1
        if mode_idx < len(sys.argv):
            mode = sys.argv[mode_idx].lower()

    if mode not in PROMPTS:
        print(f"ERROR: Unknown mode '{mode}'. Choose from: {', '.join(PROMPTS.keys())}")
        sys.exit(1)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export $(grep ANTHROPIC_API_KEY .env | xargs)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Extracting structured data")
    print(f"  Mode: {mode}")
    print(f"  Input: {input_file}")
    print(f"{'='*60}\n")

    # Read input
    text = input_file.read_text(encoding="utf-8", errors="replace")
    print(f"  Input size: {len(text):,} characters")

    # Extract
    structured = extract_structure(text, mode)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = input_file.stem
    output_file = OUTPUT_DIR / f"{safe_name}.{mode}.json"
    output_file.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Output: {output_file}")
    print(f"  Title:  {structured.get('title', 'Unknown')}")

    if mode == "curriculum":
        modules = structured.get("modules", [])
        terms = structured.get("terminology", [])
        questions = structured.get("quiz_questions", [])
        print(f"  Modules: {len(modules)}")
        print(f"  Terms:   {len(terms)}")
        print(f"  Quiz Qs: {len(questions)}")
    elif mode == "briefing":
        events = structured.get("key_events", [])
        actors = structured.get("actors", [])
        threats = structured.get("threats_and_risks", [])
        print(f"  Events:  {len(events)}")
        print(f"  Actors:  {len(actors)}")
        print(f"  Threats: {len(threats)}")
    elif mode == "guide":
        dps = structured.get("decision_points", [])
        results = structured.get("results", [])
        print(f"  Decision points: {len(dps)}")
        print(f"  Result paths:    {len(results)}")

    print()


if __name__ == "__main__":
    main()
