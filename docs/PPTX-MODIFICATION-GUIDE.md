# PPTX Modification Guide — LangGraph Workshop for National Guard

## Context for Claude Design

You are modifying an existing 18-slide PPTX presentation about building AI agents with LangGraph. The current deck is targeted at network security engineers. We need to **reframe it for a National Guard unit** with mixed backgrounds — some IT, some not. The goal is a **high-level, conceptual presentation** that uses the network automation examples to *illustrate* what AI agents can do, not to train practitioners.

**Keep the existing visual design language** (dark backgrounds, PANW branding, Cyber Orange accents, card-based layouts, code blocks in dark rounded rects). Match the typography, spacing, and slide structure already established.

**Total session time: 4 hours** (not 3.5).

---

## Summary of changes

1. **Rewrite slides 01, 02, 03** — Broaden the audience framing
2. **Insert 3 new slides** after current slide 03 (before the Agenda) — Conceptual foundation
3. **Rewrite slide 04 (Agenda)** — Update timing and add the new sections
4. **Add plain-English context cards** to code-heavy slides (06, 08, 09, 11, 13, 15)
5. **Insert 1 new slide** after current slide 17 — "Beyond network security"
6. **Update slide 18 (now slide 22)** — Broaden the closing
7. **Update all slide numbers** — New deck will be 22 slides

---

## SLIDE 01 — Title (REWRITE)

### Current
> "Building AI agents for network automation"

### New content

**Top label:**
HANDS-ON WORKSHOP / 4 HOURS

**Title:**
Building AI agents for automation

**Subtitle:**
How LangGraph, Claude, and real-world use cases show what's possible when AI reasons, decides, and acts on its own.

**Bottom info cards (3 across):**

| Card 1 | Card 2 | Card 3 |
|---------|--------|--------|
| DURATION | FORMAT | LAB |
| 4 hr | Concepts + live demos | GitHub Codespaces |
| Four sessions with breaks. Built for all backgrounds. | See the ideas, then see the code that makes them real. | Zero-install lab environment — runs in your browser. |

**Bottom attribution:**
Based on the LangGraph Workshop by Calvin Remsburg (cdot65)

**Footer:**
paloaltonetworks

---

## SLIDE 02 — Why we're here (REWRITE)

### Current
> "Network automation scripts can't reason. AI agents can."

### New content

**Section label:**
WHY WE'RE HERE

**Headline:**
Automation that can think, decide, and adapt.

**Body text:**
Traditional scripts follow a fixed recipe — step 1, step 2, step 3, done. AI agents are different. They look at a situation, decide what to do, use the right tool, check the result, and adjust. Today we'll see how that works, using network security as our example — but the patterns apply everywhere.

**Stat cards (3 across):**

| Card 1 | Card 2 | Card 3 |
|---------|--------|--------|
| DURATION | NOTEBOOKS | COST |
| 4 hr | 11 | ~$0.25/M |
| Four sessions with breaks. Conceptual first, hands-on later. | Seven in the express path, four for self-study after. | Claude Haiku tokens. Phase 1 is completely free — mock data only. |

**Slide number:** 02 / 22

---

## SLIDE 03 — Who this is for (REWRITE)

### Current
> "Network engineers who write Python — not ML researchers."

### New content

**Section label:**
WHO THIS IS FOR

**Headline:**
Anyone curious about what AI can actually do.

**Body text:**
Whether your day job is IT, logistics, operations, or something completely different — this workshop is built for you. We use network security automation as the example because it's concrete and visual, but the concepts transfer to any domain.

**"You'll get the most out of this if" section (check marks):**

| | |
|---|---|
| You're curious about AI beyond chatbots | You've heard about agents, copilots, and automation — today you'll see how they actually work under the hood. |
| You want to see real code in action | We'll walk through working examples. You don't need to write code yourself, but you'll see exactly what's happening. |
| You like learning by example | Every concept is illustrated with a concrete use case — configuring firewalls, validating settings, making decisions. |

**"You DON'T need" section (dash marks):**

| | |
|---|---|
| Programming experience | We'll explain the code as we go. Focus on the concepts, not the syntax. |
| Network security knowledge | The firewall examples are just the vehicle. We'll explain the domain as needed. |
| Any setup or accounts | GitHub Codespaces runs in your browser. Everything is pre-configured. |

**Slide number:** 03 / 22

---

## NEW SLIDE 04 — What are AI agents?

**Insert after current slide 03, before the current Agenda slide.**

**Section label:**
THE BIG IDEA

**Headline:**
What is an AI agent?

**Body text:**
An AI agent is software that can reason about a problem, choose an action, execute it, and evaluate the result — then repeat until the job is done. It's the difference between a calculator and an analyst.

**Comparison layout — two columns or two large cards:**

**LEFT CARD — "Traditional automation"**
- Follows a fixed script, every time
- Breaks when something unexpected happens
- Can't explain why it did what it did
- You have to anticipate every scenario
- Example: "If IP starts with 10., put it in the internal folder"

**RIGHT CARD — "AI agent"**
- Reads the situation and decides what to do
- Adapts when conditions change
- Can explain its reasoning
- Handles scenarios you didn't predict
- Example: "Look at this address, figure out where it belongs, validate it, and tell me what you did"

**Bottom callout:**
Today we'll build up to agents step by step — starting with the simplest possible workflow and adding intelligence one layer at a time.

**Slide number:** 04 / 22

---

## NEW SLIDE 05 — The three layers

**Section label:**
HOW IT WORKS

**Headline:**
Three layers, stacked in order.

**Body text:**
Every AI agent system has the same three layers. Today's workshop builds them from the bottom up.

**Visual — 3 stacked horizontal bars or cards, bottom to top:**

**LAYER 1 (bottom, largest) — WORKFLOW ENGINE**
Color: Use a muted base color
Label: LangGraph
Description: The plumbing. Defines what steps exist, what order they run in, and what data flows between them. This is where you build the graph — nodes, edges, conditions, loops.
Analogy: Think of it as the organizational chart for your process.

**LAYER 2 (middle) — AI BRAIN**
Color: Cortex Green accent
Label: Claude (LLM)
Description: The decision-maker. Instead of hard-coded if/else logic, an AI model reads the situation and decides what to do next. It can reason, interpret, and explain.
Analogy: This is the analyst sitting in the org chart, making judgment calls.

**LAYER 3 (top, smallest) — TOOLS**
Color: Cyber Orange accent
Label: @tool functions
Description: The hands. Python functions the AI can call — check a version, create a config, query a database. The AI picks which tool to use and when.
Analogy: These are the specific actions the analyst can take.

**Slide number:** 05 / 22

---

## NEW SLIDE 06 — Why this matters for you

**Section label:**
THE BIGGER PICTURE

**Headline:**
This isn't just about firewalls.

**Body text:**
The patterns you'll see today — workflows, decisions, loops, AI reasoning, human approval — are the same ones being used to build agents for every industry. The use case changes; the architecture doesn't.

**Grid of 6 application cards (2 rows x 3 columns):**

| Card | Domain | Example |
|------|--------|---------|
| 01 | Cybersecurity | An agent that triages alerts, queries threat intel, and drafts an incident report |
| 02 | Logistics & supply chain | An agent that checks inventory, reroutes shipments, and flags exceptions for human review |
| 03 | Intelligence analysis | An agent that ingests reports, cross-references sources, and surfaces patterns |
| 04 | IT operations | An agent that monitors systems, diagnoses issues, and executes runbooks |
| 05 | Maintenance & repair | An agent that reads sensor data, identifies failure modes, and schedules work orders |
| 06 | Mission planning | An agent that evaluates options, checks constraints, and presents COAs for approval |

**Bottom text:**
The network security examples in today's lab are a lens — not a boundary.

**Slide number:** 06 / 22

---

## SLIDE 07 — Agenda (REWRITE of current slide 04)

**Section label:**
AGENDA

**Headline:**
What we'll cover today.

**Timeline cards:**

| # | Section | Notebooks | Duration |
|---|---------|-----------|----------|
| 01 | Setting the stage — what AI agents are and why they matter | (slides only) | 20 min |
| 02 | Foundations — types, state, and your first graph | 101, 102, 103, 104 | 80 min |
| 03 | Conditional routing — decisions inside a graph | 106 | 45 min |
| 04 | AI integration — Claude, tools, and ReAct agents | 108, 110 | 60 min |
| 05 | Where this goes next — beyond network security | (slides only) | 15 min |
| 06 | Self-study path — sequential, loops, memory, HITL | 105, 107, 109, 111 | take-home |

**Slide number:** 07 / 22

---

## SLIDE 08 — Session 01 header (current slide 05)

**Only change:** Update slide number to 08 / 22. No content changes needed.

---

## SLIDE 09 — NB 101 Type annotations (current slide 06)

**Add a plain-English callout card** at the bottom or side of the existing content:

**Callout label:** PLAIN ENGLISH

**Callout text:**
Think of TypedDict as a form template. Before anyone fills it out, you define exactly which fields exist and what kind of data goes in each one. "Name" is text. "IP address" is text. "Description" is optional. If someone tries to put a number where text should go, the system catches it before anything breaks.

**Update slide number:** 09 / 22

---

## SLIDE 10 — NB 102 Core concepts (current slide 07)

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
Imagine an assembly line. The **state** is the clipboard that travels with the product. Each **node** is a workstation that reads the clipboard, does one job, and writes its results back. **Edges** are the conveyor belts connecting stations. The **graph** is the whole factory floor — you design it once, then press "go."

**Update slide number:** 10 / 22

---

## SLIDE 11 — NB 103 First graph (current slide 08)

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
This is the "hello world" — the simplest possible version. One task comes in, one workstation processes it, and the result comes out. Three pieces: a starting point, a step, and a finish line. Every complex system you'll see today is just more of these pieces snapped together.

**Update slide number:** 11 / 22

---

## SLIDE 12 — NB 104 State management (current slide 09)

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
Real jobs need more than one field on the clipboard. You need the rule name, which network zones it applies to, whether it passed validation, any errors found, how many times we've retried, and the final result. Design the clipboard first — the workstations almost design themselves.

**Update slide number:** 12 / 22

---

## SLIDE 13 — Session 02 header (current slide 10)

**Update slide number:** 13 / 22. No other changes.

---

## SLIDE 14 — NB 106 Conditional routing (current slide 11)

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
This is a decision point — like a sorting facility. A package arrives, a worker reads the label, and sends it down the right conveyor belt. The worker doesn't open the package or change anything — they just read and route. "Development config? Go left. Production config? Go right. Staging? Go to the middle lane."

**Update slide number:** 14 / 22

---

## SLIDE 15 — Session 03 header (current slide 12)

**Update slide number:** 15 / 22. No other changes.

---

## SLIDE 16 — NB 108 First LLM (current slide 13)

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
Here's where it gets interesting. Instead of a workstation running a fixed formula, we put an AI model at the station. You ask it a question in plain English, it thinks about it, and writes an answer back to the clipboard. The graph doesn't care whether the workstation runs a formula or asks an AI — it's all just "state in, state out."

**Update slide number:** 16 / 22

---

## SLIDES 17-18 — NB 110 ReAct (current slides 14-15)

**Slide 17 (ReAct pattern):**

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
This is the full loop. The AI looks at what's been asked, *thinks* about which tool would help, *uses* the tool, *reads* the result, and decides — "Am I done, or do I need another tool?" It keeps going until the job is complete. This is what people mean when they say "AI agent."

**Update slide number:** 17 / 22

**Slide 18 (Reducer + @tool):**

**Add a plain-English callout card:**

**Callout label:** PLAIN ENGLISH

**Callout text:**
Two key mechanics make this work. First: the **reducer** automatically keeps a running log of every message — what the user said, what the AI said, what the tools returned. No manual bookkeeping. Second: the **@tool decorator** turns any Python function into something the AI can discover and call. The AI reads the tool's description to decide when to use it — like reading a label on a toolbox drawer.

**Update slide number:** 18 / 22

---

## SLIDE 19 — Self-study (current slide 16)

**Update slide number:** 19 / 22. No other changes needed.

---

## SLIDE 20 — Six patterns takeaway (current slide 17)

**Update slide number:** 20 / 22. No other changes needed.

---

## NEW SLIDE 21 — Beyond network security

**Insert after current slide 17 (the "Six patterns" slide), before the closing slide.**

**Section label:**
BEYOND THE LAB

**Headline:**
Same patterns, different missions.

**Body text:**
Everything you saw today — graphs, state, routing, AI reasoning, tool use, human approval — is domain-agnostic. Here's what it looks like when you swap the use case.

**Layout: 4 scenario cards, vertically stacked or 2x2 grid. Each card has a scenario title, a brief "agent workflow" description using the patterns from the workshop, and which notebook patterns it maps to.**

**Card 1 — Cyber defense**
Scenario: A SOC analyst agent that triages incoming alerts
Workflow: Alert arrives (state) -> AI classifies severity (LLM node) -> Routes to the right playbook (conditional routing) -> Queries threat intel and enriches the alert (tool use) -> Loops until confident (looping) -> Presents findings for analyst approval (human-in-the-loop)
Patterns used: 103, 106, 108, 110, 107, 111

**Card 2 — Supply chain / logistics**
Scenario: An inventory management agent that handles reorder decisions
Workflow: Stock level triggers (state) -> AI evaluates demand forecasts (LLM node) -> Routes by priority: critical vs. routine (conditional routing) -> Checks supplier APIs and pricing (tool use) -> Drafts purchase order for human sign-off (HITL)
Patterns used: 104, 106, 108, 110, 111

**Card 3 — Intelligence analysis**
Scenario: An OSINT collection agent that cross-references sources
Workflow: Collection requirement defined (state) -> Sequential pipeline: gather, normalize, deduplicate (sequential) -> AI identifies connections and anomalies (LLM) -> Loops to gather more if confidence is low (looping) -> Produces structured summary with source citations (tool use)
Patterns used: 105, 108, 107, 110

**Card 4 — Field maintenance**
Scenario: A predictive maintenance agent for vehicle or equipment fleets
Workflow: Sensor readings ingested (state) -> AI evaluates against failure models (LLM) -> Conditional routing by risk level (routing) -> Creates work order and schedules parts (tools) -> Flags high-risk items for supervisor review (HITL)
Patterns used: 104, 108, 106, 110, 111

**Bottom text:**
The six patterns from this workshop are building blocks. The use case is up to you.

**Slide number:** 21 / 22

---

## SLIDE 22 — Closing (REWRITE of current slide 18)

### Current
> "Let's build something."

### New content

**Section label:**
WHAT'S NEXT

**Headline:**
Take it with you.

**Body text:**
The lab is yours to keep. Fork the repo, open it in Codespaces, and work through the notebooks at your own pace. No install, no cost for Phase 1, no deadline.

**Step cards (same 4-step layout as current, but updated text):**

| Step | Label | Content |
|------|-------|---------|
| 01 | FORK THE REPO | cdot65/naf-ai-agents-workshop — Click Fork to get your own copy. |
| 02 | LAUNCH CODESPACE | Click "Code" then "Create codespace on main." Two minutes to a working Jupyter Lab in your browser. |
| 03 | ADD YOUR KEY (optional) | cp .env.template .env — Get a key from console.anthropic.com. Only needed for notebooks 108-111. |
| 04 | START WITH 101 | notebooks/101_type_annotations.ipynb — Run the first cell. You're underway. |

**Bottom link:**
github.com/cdot65/naf-ai-agents-workshop

**Additional line below the link:**
Questions? Reach out anytime. These patterns are just the beginning.

**Slide number:** 22 / 22

---

## Speaker notes for all new/modified slides

### Slide 01 (Title)
"Welcome everyone. Over the next four hours, we're going to demystify AI agents — what they are, how they work, and what they can do. We'll use network security automation as our running example because it gives us concrete, visual scenarios to work with. But the patterns you'll learn today apply to any domain — cyber, logistics, intel, maintenance, you name it. No programming experience required to follow along."

### Slide 02 (Why we're here)
"You've all used automation in some form — scripts, macros, scheduled tasks. Those are great when the path is predictable. But what happens when the situation changes? When you need judgment, not just execution? That's where AI agents come in. They can reason about what's happening, pick the right tool, check their work, and adapt. That's what we're building toward today."

### Slide 03 (Who this is for)
"This workshop was originally built for network engineers, but we've adapted it for this group. Some of you work in IT during the week, some don't — and that's fine. I'm going to walk you through the concepts and show you the code in action. You don't need to understand every line. Focus on the ideas: what's the system doing, why, and what did it decide? The code is there so you can see it's real, not magic."

### Slide 04 (What are AI agents)
"Here's the core idea. On the left, traditional automation — a script that does exactly what you told it, nothing more. On the right, an AI agent — software that can look at a situation, think about it, decide what to do, do it, and then evaluate whether it worked. The agent doesn't need you to anticipate every possible scenario. It reasons its way through. That's a fundamental shift."

### Slide 05 (Three layers)
"Every agent system has these three layers. At the bottom, the workflow engine — LangGraph — which is basically the plumbing. It defines what steps exist and how data flows between them. In the middle, the AI brain — Claude — which makes decisions instead of following hard-coded rules. At the top, the tools — Python functions the AI can call to actually do things. We're going to build these layers one at a time, starting from the bottom."

### Slide 06 (Why this matters)
"Before we dive into the technical content, I want you to see the bigger picture. These six cards show different domains where the exact same patterns apply. Cybersecurity, logistics, intelligence, IT ops, maintenance, mission planning. The workflow engine doesn't care what domain it's in. The AI doesn't care whether it's analyzing a firewall rule or a supply chain bottleneck. The patterns are universal. Today's network security examples are just one lens."

### Slide 07 (Agenda)
"Here's our roadmap. We start with about 20 minutes of context-setting — that's what we've been doing. Then we spend about 80 minutes on foundations — how to define data structures, build simple workflows, and manage state. After a break, 45 minutes on decision-making inside workflows. Another break, then an hour on the exciting part — plugging in AI and watching it reason. We wrap with a look at where these patterns go beyond today's examples. There are also four take-home notebooks if you want to go deeper on your own."

### Slide 21 (Beyond network security)
"Now that you've seen how all the pieces fit together, let me show you what this looks like in other contexts. Same six patterns from slide 20, different missions. A SOC analyst agent uses all of them. A supply chain agent uses most of them. An intel analyst agent chains sequential processing with AI reasoning. A maintenance agent uses conditional routing to triage by risk. The building blocks are the same — the use case is up to you."

### Slide 22 (Closing)
"Everything we covered today is yours to keep. The repo is public, Codespaces is free for 60 hours a month, and Phase 1 doesn't cost anything — no API key needed. Fork it, open it up, work through the notebooks at your own pace. Phase 2 needs an Anthropic API key but the cost is minimal — about 25 cents per million tokens with Haiku. Questions?"

---

## Checklist for applying changes

- [ ] Rewrite slide 01 (title) — broader audience framing, 4 hours
- [ ] Rewrite slide 02 (why) — "automation that thinks" framing
- [ ] Rewrite slide 03 (who) — no prerequisites assumed
- [ ] Insert new slide 04 — "What are AI agents?"
- [ ] Insert new slide 05 — "Three layers"
- [ ] Insert new slide 06 — "Why this matters for you"
- [ ] Rewrite slide 07 (agenda) — updated timing, new sections added
- [ ] Add plain-English callout to slide 09 (NB 101)
- [ ] Add plain-English callout to slide 10 (NB 102)
- [ ] Add plain-English callout to slide 11 (NB 103)
- [ ] Add plain-English callout to slide 12 (NB 104)
- [ ] Add plain-English callout to slide 14 (NB 106)
- [ ] Add plain-English callout to slide 16 (NB 108)
- [ ] Add plain-English callout to slide 17 (NB 110 ReAct)
- [ ] Add plain-English callout to slide 18 (NB 110 reducer/tools)
- [ ] Insert new slide 21 — "Beyond network security"
- [ ] Rewrite slide 22 (closing) — broader framing
- [ ] Update all slide numbers to X / 22
- [ ] Add speaker notes to all new/modified slides
