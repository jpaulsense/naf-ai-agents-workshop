# Lab 1 — Unstructured → Structured → Output

## What You'll Build

In this lab, you'll take raw, unstructured content — Wikipedia articles, news feeds, PDFs, or transcripts — and use AI to transform it into a polished, interactive deliverable. You choose the topic. You choose the format. The AI does the heavy lifting.

**Duration:** 30-60 minutes (self-paced)
**What you need:** Your Codespace terminal and a topic you care about

## What You'll Learn

- How to fetch content from multiple source types (APIs, feeds, documents)
- The difference between raw content and structured data
- How to use AI to extract meaning and organize information
- How to generate interactive outputs from structured data
- The real-world pattern: **Fetch → Structure → Create**

```
FETCH  →  STRUCTURE  →  CREATE
(get raw content)  (AI extracts meaning)  (build something useful)
```

This three-step pattern is behind almost every useful AI application. The same steps apply whether you're building training curricula, intelligence briefings, decision guides, compliance checklists, or operational summaries.

---

## The Pipeline

You have three scripts that form a pipeline. Each one does one job:

```
scripts/fetch_source.py      → Gets raw text from a source
scripts/extract_structure.py → Sends text to Claude, gets structured JSON
scripts/build_output.py      → Renders JSON into interactive HTML
```

You can run them one at a time, or chain them together. By the end, you'll have an interactive HTML file you can open in your browser.

---

## Step 1: Choose Your Source (5 min)

Pick a topic and a source type. Here are your options:

### Wikipedia — Any topic you can think of

```bash
python scripts/fetch_source.py wikipedia "Incident Command System"
python scripts/fetch_source.py wikipedia "Drone warfare"
python scripts/fetch_source.py wikipedia "Supply chain management"
python scripts/fetch_source.py wikipedia "Cybersecurity"
```

### RSS News Feeds — Current events

```bash
python scripts/fetch_source.py rss "https://feeds.bbci.co.uk/news/world/rss.xml"
python scripts/fetch_source.py rss "https://feeds.bbci.co.uk/news/technology/rss.xml"
python scripts/fetch_source.py rss "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
```

### PDF Documents — Government docs, manuals, regulations

```bash
python scripts/fetch_source.py pdf "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf"
```

Any publicly accessible PDF URL works. Try finding documents relevant to your job — regulations, frameworks, policy documents.

### Local Files — Pre-loaded transcripts

```bash
python scripts/fetch_source.py file "transcripts/01-aircAruvnKk.md"
python scripts/fetch_source.py file "transcripts/05-LPZh9BOjkQs.md"
```

The `transcripts/` folder has 9 pre-loaded video transcripts on neural networks and AI.

---

After running `fetch_source.py`, your raw content is saved in `lab1_output/raw/`. Check what you got:

```bash
ls lab1_output/raw/
```

---

## Step 2: Choose Your Output Mode (2 min)

Now decide what you want to BUILD from your source material. You have three options:

| Mode | Best for | What you get |
|------|----------|--------------|
| `curriculum` | Training material, study guides | Modules, terminology, study guide, interactive quiz |
| `briefing` | News, intelligence, situation reports | Timeline, actors, threats, recommendations |
| `guide` | Manuals, procedures, regulations | Interactive decision tree with branching paths |

---

## Step 3: Extract Structure (2-3 min)

Run the extraction script on your raw text file. Replace the filename with whatever `fetch_source.py` created in `lab1_output/raw/`:

```bash
# For training material:
python scripts/extract_structure.py lab1_output/raw/YOUR_FILE.txt --mode curriculum

# For a news briefing:
python scripts/extract_structure.py lab1_output/raw/YOUR_FILE.txt --mode briefing

# For an interactive decision guide:
python scripts/extract_structure.py lab1_output/raw/YOUR_FILE.txt --mode guide
```

This sends your content to Claude AI, which extracts structured data and returns it as JSON. The output is saved in `lab1_output/structured/`.

Want to see what the AI extracted? Look at the JSON:

```bash
cat lab1_output/structured/YOUR_FILE.curriculum.json | python -m json.tool | head -50
```

---

## Step 4: Build Your Output (instant)

Now turn that structured JSON into an interactive HTML page:

```bash
python scripts/build_output.py lab1_output/structured/YOUR_FILE.curriculum.json
```

The script auto-detects the mode from the filename and uses the right template.

Your HTML file is in `lab1_output/final/`. To view it:

```bash
python -m http.server 8000 --directory lab1_output/final
```

Then click the port 8000 link in the Codespace ports panel (or the pop-up notification). Your interactive output opens in the browser.

---

## Step 5: Iterate (10-20 min)

You've completed one full pipeline run. Now try variations:

### Try a different mode on the same source

```bash
# Same source, different output format
python scripts/extract_structure.py lab1_output/raw/YOUR_FILE.txt --mode briefing
python scripts/build_output.py lab1_output/structured/YOUR_FILE.briefing.json
```

### Combine multiple sources

Fetch several related articles or documents, then concatenate them before structuring:

```bash
python scripts/fetch_source.py wikipedia "Army physical fitness"
python scripts/fetch_source.py wikipedia "Military training"
cat lab1_output/raw/Army_physical_fitness.txt lab1_output/raw/Military_training.txt > lab1_output/raw/combined_fitness.txt
python scripts/extract_structure.py lab1_output/raw/combined_fitness.txt --mode curriculum
python scripts/build_output.py lab1_output/structured/combined_fitness.curriculum.json
```

### Use Claude Code to help

Open Claude Code in your terminal and ask it to help you build something more complex:

```bash
claude
```

Try prompts like:
- "Help me fetch 3 Wikipedia articles about drone operations and combine them into a training curriculum"
- "I want to build a decision guide from this PDF — can you help me refine the output?"
- "The quiz questions in my curriculum aren't great — can you improve them?"

Claude Code can read your files, run the scripts, and iterate with you.

---

## Step 6: Reflect (5 min)

Before you move on, think about these questions:

1. **What surprised you about the AI's output?** Did it organize the information the way you would have? What would you change?

2. **What did it get wrong?** AI makes mistakes — misinterpreted facts, invented details, poor sequencing. What did you catch? This is why humans stay in the loop.

3. **What would you build for your actual job?** The fetch → structure → create pattern applies to any unstructured content:
   - Fetch incident reports → structure by category → create trend analysis
   - Fetch regulations → structure into requirements → create compliance checklist
   - Fetch maintenance logs → structure by system → create readiness summary
   - Fetch after-action reviews → structure into lessons learned → create training update
   - Fetch multiple SOPs → structure decision points → create interactive field guide

4. **What's the value?** If the AI output is 80% there and needs 20 minutes of cleanup — that's a win. Without AI, building this from scratch takes hours or days. The value is in speed, not blind trust.

---

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
```bash
export $(grep ANTHROPIC_API_KEY .env | xargs)
```

**"No module named 'pymupdf'" (or any module)**
```bash
source .venv/bin/activate
uv sync
```

**PDF download times out**
Some large PDFs are slow from cloud environments. Try a smaller document, or use Wikipedia/RSS instead.

**Claude returns invalid JSON**
This happens occasionally. Just re-run the `extract_structure.py` command — the temperature is set to 0 so results are deterministic, but network issues can truncate responses.

**"I want to use Claude Code but it's not installed"**
```bash
sudo npm install -g @anthropic-ai/claude-code
export $(grep ANTHROPIC_API_KEY .env | xargs)
claude
```

**"I'm stuck or confused"**
Raise your hand — the proctor is here to help. Or ask Claude Code: `claude "I'm stuck on step 3 of the lab, help me"`

---

## What's Next

This lab established the **fetch → structure → create** pattern. In the LangGraph notebooks that follow, you'll learn to build AI agents that run this same pipeline *autonomously* — deciding which tool to use, when to use it, and when the task is done.

| This lab (manual) | Agent equivalent (automated) |
|-------------------|------------------------------|
| You run `fetch_source.py` | An agent calls a tool to fetch data |
| You run `extract_structure.py` | An agent calls an LLM node to reason |
| You review the JSON | Human-in-the-loop approval step |
| You run `build_output.py` | An agent produces final output |
| You iterate with Claude Code | An agent loops until quality threshold met |

The difference between a tool and an agent: a tool does one thing when you tell it. An agent decides *what* to do, *when* to do it, and *whether it's done* — on its own.

---

## Quick Reference: All Commands

```bash
# FETCH — get raw content
python scripts/fetch_source.py wikipedia "<topic>"
python scripts/fetch_source.py rss "<feed_url>"
python scripts/fetch_source.py pdf "<pdf_url>"
python scripts/fetch_source.py file "<local_path>"

# STRUCTURE — extract meaning with AI
python scripts/extract_structure.py <raw_file> --mode curriculum
python scripts/extract_structure.py <raw_file> --mode briefing
python scripts/extract_structure.py <raw_file> --mode guide

# CREATE — build interactive output
python scripts/build_output.py <structured_json_file>

# VIEW — open in browser
python -m http.server 8000 --directory lab1_output/final

# HELP — ask Claude Code
claude
```
