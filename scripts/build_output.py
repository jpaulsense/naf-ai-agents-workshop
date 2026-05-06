"""
Build interactive HTML output from structured JSON data.

Takes the JSON output from extract_structure.py and renders it into
a self-contained HTML file using Jinja2 templates.

Usage:
    python scripts/build_output.py lab1_output/structured/MyTopic.curriculum.json
    python scripts/build_output.py lab1_output/structured/BBC_World.briefing.json
    python scripts/build_output.py lab1_output/structured/NIST_CSF.guide.json

Output is saved to lab1_output/final/ as a self-contained .html file.
"""

import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path("lab1_output/final")


def detect_mode(data: dict, filename: str) -> str:
    """Detect the output mode from the JSON structure or filename."""
    # Try filename first (e.g., "topic.curriculum.json")
    parts = filename.rsplit(".", 2)
    if len(parts) >= 3:
        mode = parts[-2]
        if mode in ("curriculum", "briefing", "guide"):
            return mode

    # Fall back to detecting by structure
    if "decision_points" in data:
        return "guide"
    elif "key_events" in data or "threats_and_risks" in data:
        return "briefing"
    else:
        return "curriculum"


def build_curriculum(data: dict) -> str:
    """Render curriculum data to HTML."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("curriculum.html")
    return template.render(
        title=data.get("title", "Training Curriculum"),
        summary=data.get("summary", ""),
        modules=data.get("modules", []),
        terminology=data.get("terminology", []),
        key_facts=data.get("key_facts", []),
        common_misconceptions=data.get("common_misconceptions", []),
        quiz_questions=data.get("quiz_questions", []),
    )


def build_briefing(data: dict) -> str:
    """Render briefing data to HTML."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("briefing.html")
    return template.render(
        title=data.get("title", "Situation Briefing"),
        classification=data.get("classification", "UNCLASSIFIED"),
        date_produced=data.get("date_produced", ""),
        summary=data.get("summary", ""),
        key_events=data.get("key_events", []),
        actors=data.get("actors", []),
        threats_and_risks=data.get("threats_and_risks", []),
        trends=data.get("trends", []),
        recommendations=data.get("recommendations", []),
        information_gaps=data.get("information_gaps", []),
        sources_used=data.get("sources_used", ""),
    )


def build_guide(data: dict) -> str:
    """Render decision guide data to HTML."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("guide.html")
    return template.render(
        title=data.get("title", "Decision Guide"),
        purpose=data.get("purpose", ""),
        terminology=data.get("terminology", []),
        quick_reference=data.get("quick_reference", []),
        data_json=json.dumps(data),
    )


BUILDERS = {
    "curriculum": build_curriculum,
    "briefing": build_briefing,
    "guide": build_guide,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

USAGE = """
Usage: python scripts/build_output.py <structured_json_file>

The output mode (curriculum, briefing, or guide) is detected automatically
from the filename or JSON structure.

Examples:
    python scripts/build_output.py lab1_output/structured/AI.curriculum.json
    python scripts/build_output.py lab1_output/structured/News.briefing.json
    python scripts/build_output.py lab1_output/structured/Manual.guide.json
"""


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    # Load JSON
    data = json.loads(input_file.read_text(encoding="utf-8"))

    if "raw_response" in data:
        print("ERROR: The input file contains a raw (unparsed) response from Claude.")
        print("       Re-run extract_structure.py or manually fix the JSON.")
        sys.exit(1)

    # Detect mode
    mode = detect_mode(data, input_file.name)
    builder = BUILDERS[mode]

    print(f"\n{'='*60}")
    print(f"  Building {mode} output")
    print(f"  Input: {input_file}")
    print(f"{'='*60}\n")

    # Build HTML
    html = builder(data)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = input_file.stem.rsplit(".", 1)[0]  # Remove the mode suffix
    output_file = OUTPUT_DIR / f"{safe_name}_{mode}.html"
    output_file.write_text(html, encoding="utf-8")

    print(f"  Output: {output_file}")
    print(f"  Size:   {len(html):,} bytes")
    print(f"  Mode:   {mode}")
    print(f"\n  Open in browser to view:")
    print(f"  python -m http.server 8000 --directory lab1_output/final")
    print(f"  Then visit: http://localhost:8000/{output_file.name}")
    print()


if __name__ == "__main__":
    main()
