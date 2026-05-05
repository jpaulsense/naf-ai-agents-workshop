# Lab 1 — YouTube → Training Curriculum Builder

## What You'll Build

In this lab, you'll use AI to turn YouTube video content into a complete training curriculum — study guide, lesson sequence, and quiz included. You'll do it in about 30 minutes using a pattern that would take days to do manually.

**Duration:** 30-45 minutes (self-paced)
**What you need:** A browser with [claude.ai](https://claude.ai) or [chat.openai.com](https://chat.openai.com) open

## What You'll Learn

- How AI search works differently from Google (semantic/meaning-based vs. keyword)
- The difference between raw content and structured data
- How to prompt AI to transform structured data into useful outputs
- The real-world pattern: **Fetch → Structure → Create**

```
FETCH  →  STRUCTURE  →  CREATE
(get content)  (extract key info)  (build something useful)
```

This three-step pattern is behind almost every useful AI application. YouTube videos are today's example, but the same steps apply to any unstructured content — incident reports, regulations, meeting notes, maintenance logs.

---

## Step 1: Choose Your Path (2 min)

This lab gives you two options for getting source material:

### Option A: Use Pre-Loaded Transcripts (recommended)

The `transcripts/` folder in this repository contains 9 pre-extracted video transcripts from the 3Blue1Brown "Neural Networks" series. These cover:

| File | Topic |
|------|-------|
| `01-aircAruvnKk.md` | What is a neural network? |
| `02-IHZwWFHWa-w.md` | Gradient descent |
| `03-Ilg3gGewQ5U.md` | Backpropagation |
| `04-tIeHLnjs5U8.md` | Backpropagation calculus |
| `05-LPZh9BOjkQs.md` | GPT (Generative Pre-trained Transformer) |
| `06-wjZofJX0v4M.md` | Attention in transformers |
| `07-eMlx5fFNoYc.md` | How LLMs might store facts |
| `08-9-Jl0dxWQs8.md` | Transformers (visual intro) |
| `09-iv-5mZ_9CPY.md` | Word embeddings |

**To use these:** Open 3-4 transcript files that interest you. You'll paste their content into Claude/ChatGPT in Step 3.

### Option B: Run the Extraction Script (advanced)

If you have an `ANTHROPIC_API_KEY` configured (check your `.env` file), you can extract transcripts from any YouTube videos:

1. Edit `videos.txt` — add 1-5 YouTube URLs (one per line)
2. Run: `python scripts/youtube_extract.py videos.txt`
3. Find your structured output in `lab1_output/`

This uses the YouTube Transcript API to fetch captions, then Claude AI to extract structured knowledge. It produces both JSON (discrete facts) and YAML (contextual narratives).

> **Note:** Option B requires an API key and takes 2-3 minutes to run. If you're unsure, start with Option A — you can always try Option B later.

---

## Step 2: Understand the Structure (5 min)

Before you build your curriculum, take a moment to understand *why* structured data matters.

Open one of the transcript files (e.g., `transcripts/01-aircAruvnKk.md`). Scroll through it. Notice:
- It's a wall of timestamped text
- Finding specific facts requires reading the whole thing
- There's no organization — topics blend together

Now imagine trying to build a training curriculum from 4 of these raw transcripts. You'd have to:
1. Read all of them end to end
2. Mentally extract the key concepts
3. Figure out how they relate to each other
4. Organize them into a logical sequence
5. Write up summaries, definitions, and quiz questions

That's hours of work. Instead, you're going to hand this to AI and let it do steps 1-4 in seconds. Your job is step 5: reviewing, correcting, and approving the output.

**This is the core insight:** AI is a *tool*, not a replacement. It does 90% of the work in 30 seconds, but you still need a human to review, correct, and approve. The value is in the speed — not in blind trust.

---

## Step 3: Build Your Curriculum (15-20 min)

This is the main event. You'll paste transcript content into Claude or ChatGPT along with a structured prompt, and it will generate a complete curriculum package.

### 3.1 — Gather your source material

Pick 3-4 transcripts from the `transcripts/` folder (or use your `lab1_output/` files from Option B). Copy the full content of each.

### 3.2 — Open your AI tool

Go to [claude.ai](https://claude.ai) or [chat.openai.com](https://chat.openai.com) in your browser.

### 3.3 — Paste this prompt

Copy the entire block below into your AI chat. Replace `[TOPIC]` with your actual topic (e.g., "neural networks and deep learning"). Then paste your transcript content where indicated at the bottom.

```
I have transcripts from several YouTube videos on [TOPIC].
Using ONLY the content from these transcripts, create a training
curriculum package:

## 1. CURRICULUM OUTLINE
- Organize the content into 3-5 logical modules, sequenced from
  foundational to advanced
- Each module should have: a title, learning objective (one sentence),
  and 3-5 key topics covered
- Note which video(s) each module draws from

## 2. ONE-PAGE STUDY GUIDE
- Key definitions and terminology (bulleted list)
- Core concepts explained in 1-2 sentences each
- "Common misconceptions" section (things people get wrong)
- "Key takeaways" — the 5 most important things to remember

## 3. KNOWLEDGE CHECK (10 questions)
- Mix of multiple choice (7) and short answer (3)
- Include the correct answer and a brief explanation for each
- Questions should test understanding, not memorization
- Range from basic recall to applied scenarios

## 4. SOURCES
- List each video used: title and video ID
- Note which curriculum modules each video contributed to

Format everything in clean markdown with clear headers.

---

VIDEO TRANSCRIPTS:

[PASTE YOUR TRANSCRIPTS HERE]
```

### 3.4 — Review the output

Once the AI generates your curriculum, read through it critically:

- **Accuracy:** Did it get the facts right? Are definitions correct?
- **Completeness:** Did it miss any major concepts from the videos?
- **Sequence:** Does the module order make sense? Would a learner follow this progression?
- **Quiz quality:** Do the questions test real understanding, or just surface-level recall?
- **Hallucinations:** Did it add information that *wasn't* in the transcripts?

Mark anything that looks wrong or suspicious. This is the "human in the loop" step.

---

## Step 4: Iterate and Improve (5-10 min)

Your first output probably isn't perfect. Here are some follow-up prompts to try:

**If the output is too generic:**
> "Be more specific — use exact terminology, examples, and details from the video transcripts. Don't add information that isn't in the sources."

**If the study guide is too short:**
> "Expand the study guide to include at least 15 key terms and 10 core concepts. Pull directly from the transcript content."

**If you suspect hallucinated content:**
> "Review your output against the source transcripts. Flag any claims that aren't directly supported by the video content, and remove them."

**If you want a different format:**
> "Reformat the curriculum as a 1-week training plan with daily objectives, reading assignments (video segments), and daily quizzes of 3 questions each."

**If you want to go deeper on one module:**
> "Expand Module [X] into a detailed 30-minute lesson plan with: an opening hook, 3 teaching points with examples, a practice activity, and a summary."

---

## Step 5: Reflect (5 min)

Before you move on, think about these questions:

1. **Would you actually use this output?** If it's 80% there and needs 20 minutes of cleanup — that's a win. Without AI, building a curriculum from scratch takes hours or days.

2. **What did the AI get wrong?** Every AI makes mistakes — misinterpreting a video's point, inventing details, or sequencing topics oddly. What did you catch?

3. **What else could you build with this same pattern?** The fetch → structure → create pattern applies to any unstructured content:
   - Fetch incident reports → structure into categories → create a trend analysis
   - Fetch regulations → structure into requirements → create a compliance checklist
   - Fetch meeting notes → structure into decisions/action items → create a status report
   - Fetch maintenance logs → structure by system → create a readiness summary

4. **How does this connect to what you learned in Session 0?** Everything that happened in this lab uses the same architecture from the morning session:
   - The transcript extraction uses tokenization (breaking text into chunks)
   - The AI's ability to understand context uses attention (weighing which words relate to which)
   - The curriculum generation uses next-token prediction (the model predicts the most useful next word, one at a time)

---

## Bonus: Run the Structured Extraction Script

If you finished early and want to see how the automated pipeline works, try running the full extraction script:

```bash
# Edit videos.txt with YouTube URLs you're interested in
# (one URL per line, max 5 videos)

python scripts/youtube_extract.py videos.txt
```

This script:
1. Fetches transcripts via the YouTube Transcript API
2. Sends each transcript to Claude AI for structured extraction
3. Produces two complementary knowledge formats:
   - **JSON** — discrete facts, terminology, statistics (good for search, quizzes)
   - **YAML** — narrative knowledge, arguments, connections (good for AI agents, conversations)
4. Combines everything into a knowledge base with a human-readable summary

Look at the output in `lab1_output/` and compare the JSON vs YAML formats. Notice how the same video content gets represented differently depending on the use case.

---

## Troubleshooting

**"I don't have access to Claude or ChatGPT"**
- Claude free tier: [claude.ai](https://claude.ai) — sign up with Google or email
- ChatGPT free tier: [chat.openai.com](https://chat.openai.com) — sign up with Google or email
- Both work for this lab. Use whichever you can access.

**"The output is too short / not detailed enough"**
- Make sure you pasted enough transcript content. 3-4 full transcripts gives the AI more to work with.
- Add to your prompt: "Be comprehensive and detailed. Use specific examples from the transcripts."

**"The AI seems to be making things up"**
- Add to your prompt: "Only include information explicitly stated in the transcripts. If you're unsure whether something was covered, leave it out and note the gap."
- This is called "hallucination" — it's a known limitation. Your job as the human is to catch it.

**"The youtube_extract.py script isn't working"**
- Check that your `.env` file has `ANTHROPIC_API_KEY` set
- Make sure you've run `uv sync` to install dependencies
- Try with just 1 video URL first to isolate the issue

**"I'm stuck or confused"**
- Raise your hand — the proctor is here to help
- Pair up with someone nearby and work through it together

---

## What's Next

This lab established the **fetch → structure → create** pattern. In the LangGraph notebooks that follow, you'll learn to build AI agents that run this same loop *autonomously* — deciding which tool to use, when to use it, and when the task is done.

| This lab (manual) | Agent equivalent (automated) |
|-------------------|------------------------------|
| You copy transcripts | An agent fetches from an API |
| You paste into Claude | An agent calls an LLM node |
| You review the output | Human-in-the-loop approval |
| You iterate with follow-ups | An agent loops until quality threshold is met |

The difference between a tool and an agent: a tool does one thing when you ask. An agent decides *what* to do, *when* to do it, and *whether it's done* — on its own.
