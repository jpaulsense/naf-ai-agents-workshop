# Lab 1 — YouTube → Training Curriculum Builder

## Overview

**Duration:** 45-60 minutes
**Position in schedule:** Day 1, after Session 0 (What Is AI) and the break — first hands-on activity
**Audience:** Non-technical; mixed backgrounds
**Prerequisites:** A browser, access to Claude (claude.ai), and the facilitator's Open Brain YouTube tools running via MCP

**What they'll build:** Each participant picks a topic they care about, uses AI to search and summarize YouTube videos on that topic, then uses the structured data to generate a complete training curriculum — study guide, knowledge check, and all.

**What they'll learn:**
- How AI search works differently from Google (semantic/meaning-based vs. keyword)
- The difference between raw content and structured data
- How to prompt AI to transform structured data into useful outputs
- The real-world pattern: **fetch → structure → create**

---

## Lab Setup

### What the facilitator needs running

This lab uses the Open Brain MCP tools. The facilitator (you) will be driving the tool calls from your Claude Code / Claude Desktop session with MCP connected. Participants follow along on the projector for Steps 1-3, then break out to their own Claude sessions for Step 4.

**Required MCP tools:**
- `youtube_search` — semantic search across indexed video summaries
- `youtube_summary` — pull full structured summary for a specific video
- `youtube_note` — save annotations (used in bonus step)

**Participant requirements:**
- A laptop or phone with a browser
- Access to Claude (claude.ai free tier is fine) or ChatGPT
- No accounts, API keys, or installs needed for their part

### Facilitator pre-check

Before the lab, verify the YouTube tools are responding:

```
Test: youtube_search query="cybersecurity basics" limit=3
Expected: Returns 3 video results with titles and summaries
```

---

## Lab Flow

### Step 0: Facilitator Introduction (5 min)

**What to say:**

> "OK, we just spent two hours learning how AI works under the hood. Now let's actually use it. We're going to build something real — a training curriculum — using AI at every step.
>
> Here's the pattern we're going to follow. It's three steps, and it's the same pattern behind almost every useful AI application:
>
> 1. **Fetch** — Get raw content from somewhere (in our case, YouTube videos)
> 2. **Structure** — Use AI to extract the key concepts, facts, and takeaways into organized data
> 3. **Create** — Use AI to transform that structured data into a finished product
>
> By the end of this lab, each of you will have a training curriculum on a topic you actually care about — with a study guide, a lesson sequence, and a quiz. And you'll have built it in about 30 minutes using a system that would have taken days to do manually.
>
> I'm going to drive the first two steps up here on the screen so you can see how the tools work. Then you'll break out and do Step 3 on your own laptops."

**Put this on the screen:**

```
FETCH  →  STRUCTURE  →  CREATE
(search)   (summarize)   (build curriculum)
```

---

### Step 1: Pick a Topic & Search (10 min)

**Facilitator-led, projected on screen**

**What to say:**

> "First, we need a topic. I need a volunteer — give me something you'd want to train your team on. Could be anything: leadership, cybersecurity, drone operations, first aid, vehicle maintenance, public speaking, AI itself — whatever you're interested in."

Take a suggestion from the audience. For this guide, we'll use **"cybersecurity basics"** as the example, but use whatever the group picks.

**Run the search live on screen:**

```
youtube_search query="cybersecurity basics for beginners" limit=10
```

**What to say while results load:**

> "What I just did is a *semantic search* — not a keyword search. I didn't search for pages that contain the exact words 'cybersecurity basics.' I searched for videos whose *meaning* is about cybersecurity basics. The AI is using those embeddings we talked about in Session 0 — it converted my query into a vector and found videos whose summary vectors are nearby in that high-dimensional space. So it might return a video titled 'How Hackers Actually Attack Networks' even though those words don't match my search — because the *meaning* matches."

**When results appear, talk through them:**

> "OK, we got [X] results back. Let me read through the titles and descriptions. [Read through 3-5 of the most relevant ones.] These look like good candidates. Now let's pick 3-4 of the best ones and pull their full summaries. That's Step 2 — structuring."

**Facilitator tip:** If the search returns too few or irrelevant results, try a broader or different query. This is a good teaching moment — "see, even AI search requires iteration. The first query isn't always the best. Let's try rephrasing."

---

### Step 2: Pull Structured Summaries (10 min)

**Facilitator-led, projected on screen**

Pick 3-4 videos from the search results. For each one, pull the full summary:

```
youtube_summary video_id="[VIDEO_ID_1]"
youtube_summary video_id="[VIDEO_ID_2]"
youtube_summary video_id="[VIDEO_ID_3]"
```

**What to say while the first one loads:**

> "Now I'm pulling the full structured summary for each video. This isn't just the title and description — the system has already watched these videos and extracted structured data: key topics, main arguments, technical concepts, takeaways. This is the AI doing that 'fetch → structure' step — taking raw unstructured video content and turning it into organized, searchable data.
>
> This is the same concept we talked about in Session 0 with the MLP layers — the model is extracting facts and relationships from the content and encoding them as structured information."

**When summaries come back, highlight the structure:**

> "Look at what came back. We have:
> - A summary of the video's main points
> - Key topics and concepts
> - Takeaways
> - Technical depth rating
>
> This is *structured data* — not a wall of text, but organized information with defined fields. That's what makes the next step possible. You can't easily build a curriculum from a 20-minute video transcript. But you *can* build one from a clean summary with extracted topics and takeaways."

**Compile the summaries** — copy all 3-4 summaries into a single text block. This becomes the input for Step 3.

---

### Step 3: Build the Curriculum (15-20 min)

**Participant-led — everyone on their own laptops**

This is where participants break out. They'll paste the compiled summaries into their own Claude or ChatGPT session and use a prompt to generate the curriculum.

**Put this on the screen and have them copy it:**

---

#### Prompt Template (display on screen for participants to copy)

```
I have structured summaries from several YouTube videos on [TOPIC].
Using ONLY the content from these summaries, create a training
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
- List each video used: title, channel, and video ID
- Note which curriculum modules each video contributed to

Format everything in clean markdown with clear headers.

---

VIDEO SUMMARIES:

[PASTE THE COMPILED SUMMARIES HERE]
```

---

**What to say when handing off:**

> "OK, here's where you take over. On the screen is a prompt template. You're going to:
>
> 1. Open Claude or ChatGPT on your laptop — free accounts work fine
> 2. Copy this prompt
> 3. Replace [TOPIC] with the topic we searched for
> 4. Paste the video summaries where it says [PASTE HERE] — I'm going to send those to you now [share via chat, email, or just leave them on screen to copy]
> 5. Hit enter and watch it build your curriculum
>
> You'll get back a full curriculum outline, a study guide, and a 10-question quiz — all generated from real video content. Take about 10-15 minutes. Read through what it produces. Does it make sense? Did it get anything wrong? Would you actually use this to train someone?
>
> When you're done, we'll regroup and compare notes."

**Facilitator:** While participants work, walk the room. Help anyone who's stuck. Look for interesting outputs to highlight in the debrief.

---

### Step 4: Debrief & Discuss (10 min)

**Facilitator-led, full group**

Bring the group back together. Ask 2-3 participants to share highlights.

**Discussion questions:**

1. **"Did the AI get anything wrong?"**
   > This is the most important question. AI makes mistakes — it can misinterpret a video's point, invent details that weren't in the summary, or sequence topics in a way that doesn't make sense. Talk about what they caught.
   >
   > "This is why we say AI is a *tool*, not a replacement. It did 90% of the work in 30 seconds, but you still need a human to review, correct, and approve. The value is in the speed — not in blind trust."

2. **"Would you actually use this output?"**
   > Get honest reactions. Some will say yes, some will say it needs work. Both are valid.
   >
   > "If the answer is 'it's 80% there and I'd need 20 minutes to clean it up' — that's a win. Without AI, building a curriculum from scratch takes hours or days. Even imperfect AI output is a massive accelerator."

3. **"What else could you build with this same pattern?"**
   > Push them to generalize the fetch → structure → create pattern:
   > - Fetch incident reports → structure into categories → create a trend analysis
   > - Fetch regulations → structure into requirements → create a compliance checklist
   > - Fetch meeting notes → structure into decisions and action items → create a status report
   > - Fetch maintenance logs → structure by system → create a readiness summary
   >
   > "The *pattern* is what matters. YouTube videos are just today's example. The same three steps apply to any unstructured content."

4. **"How does this connect to what we learned in Session 0?"**
   > Tie it back to the technical concepts:
   > - The **semantic search** used embeddings and dot products — the same vector math we covered
   > - The **summary extraction** used attention (understanding context) and MLPs (extracting facts)
   > - The **curriculum generation** used next-token prediction — the model predicted the most likely curriculum structure word by word
   > - The entire pipeline is running on the same architecture: tokens → embeddings → attention → MLPs → prediction
   >
   > "Everything you used in this lab is powered by the nine concepts from this morning. Neurons, layers, weights, gradient descent, embeddings, attention, MLPs — all of it, working together."

---

### Bonus Step (if time allows): Save & Annotate

If you have 5 extra minutes, demonstrate the "memory" capability:

```
youtube_note video_id="[BEST_VIDEO_ID]" note="Used in cybersecurity basics curriculum lab — best video for explaining phishing attacks to non-technical audience"
```

**What to say:**

> "One more thing. The system I used to search and summarize those videos also has memory. I can annotate a video — save a note about *why* it was useful and *how* I used it. Next time someone asks me for cybersecurity training content, the system remembers that this video was good for non-technical audiences. It learns from use.
>
> This is the difference between a tool and a *system*. A tool does one thing. A system learns and improves over time. That 'note' I just saved? It's stored as an embedding — the same vector representation we talked about this morning — and it'll influence future searches and recommendations."

---

## Facilitator Cheat Sheet

### If youtube_search returns no results or poor results

Try these alternative queries:
- Broaden the topic: "network security" instead of "cybersecurity for beginners"
- Use different framing: "how hackers work" instead of "cybersecurity"
- Try a different topic entirely — have a backup topic ready

### If a participant's Claude session gives a weak curriculum

Common issues and fixes:
- **Too generic:** Tell them to add to the prompt: "Be specific — use exact terminology, examples, and details from the video summaries. Don't add information that isn't in the sources."
- **Too short:** Add: "Make the study guide comprehensive — at least 15 key terms and 10 core concepts."
- **Hallucinated content:** Add: "Only include information explicitly stated in the video summaries. If you're unsure whether something was covered, leave it out and note the gap."

### Backup plan: pre-loaded summaries

If the MCP tools are down or unresponsive, have 3-4 pre-pulled video summaries saved as a text file. Distribute them manually and skip straight to Step 3. The learning is in the prompting and output evaluation, not in watching the API calls.

### Timing guide

| Step | Activity | Duration | Running total |
|------|----------|----------|---------------|
| 0 | Intro & pattern explanation | 5 min | 0:05 |
| 1 | Topic selection & search | 10 min | 0:15 |
| 2 | Pull & review summaries | 10 min | 0:25 |
| 3 | Participants build curriculum | 15-20 min | 0:40-0:45 |
| 4 | Debrief & discussion | 10 min | 0:50-0:55 |
| Bonus | Save & annotate demo | 5 min | 0:55-1:00 |

---

## What This Lab Sets Up

This lab establishes the **fetch → structure → create** pattern that recurs throughout the rest of the workshop:

| This lab | Agent workshop equivalent |
|----------|--------------------------|
| `youtube_search` | An agent querying a database or API |
| `youtube_summary` | An agent extracting structured data from a source |
| Claude prompt → curriculum | An agent using an LLM node to reason and generate |
| Human reviews output | Human-in-the-loop approval (Notebook 111) |

When you transition to the LangGraph workshop after this lab, you can reference back:

> "Remember the curriculum builder? You did fetch → structure → create manually — you ran each step yourself. In the agent workshop, we're going to teach the AI to run that loop *on its own* — deciding which tool to use, when to use it, and when it's done. That's what makes it an agent."

---

## Sample Output (for facilitator reference)

Below is an example of what a good curriculum output looks like, so you know what to expect and can coach participants toward it:

### Example: Cybersecurity Basics Curriculum

**Module 1: What Is Cybersecurity? (Foundational)**
- Learning objective: Understand what cybersecurity is, why it matters, and the basic threat landscape
- Topics: Definition of cybersecurity, types of threats (malware, phishing, ransomware), the CIA triad (confidentiality, integrity, availability), who attackers are and what motivates them
- Source videos: Video 1, Video 3

**Module 2: How Attacks Actually Work (Intermediate)**
- Learning objective: Understand the mechanics of common attack types
- Topics: Phishing anatomy (email → click → payload), social engineering techniques, network-based attacks (man-in-the-middle, DNS spoofing), the attack kill chain
- Source videos: Video 2, Video 4

**Module 3: Defending Yourself and Your Organization (Applied)**
- Learning objective: Apply practical security measures in daily work
- Topics: Password hygiene and MFA, recognizing phishing attempts, safe browsing habits, reporting procedures, physical security basics
- Source videos: Video 1, Video 3, Video 4

**Study Guide:** [15-20 key terms with definitions, 8-10 core concepts, 3-5 common misconceptions]

**Knowledge Check:** [10 questions mixing recall ("What does CIA stand for in cybersecurity?") with application ("Your colleague receives an email from IT asking them to click a link to reset their password. The email came from it-support@company-secure.net instead of the usual @company.com. What should they do?")]
