# Complete Lab Guide — AI Agents Workshop

## Two-Day Workshop for National Guard Unit

**Duration:** 8 hours total (4 hours per day)
**Audience:** Non-technical; mixed backgrounds
**Environment:** GitHub Codespaces (browser-based, zero install) + Claude AI (claude.ai)

---

## Schedule Overview

### Day 1 (4 hours)

| Time | Activity | Type |
|------|----------|------|
| 0:00 | Session 0: What Is AI and How Does It Work? | Presentation |
| 2:00 | Break | 15 min |
| 2:15 | **Lab 1: YouTube → Training Curriculum Builder** | Hands-on |
| 3:15 | **Lab 2: Your First AI Workflow (Notebooks 101-103)** | Hands-on |
| 3:55 | Day 1 Wrap-up | 5 min |

### Day 2 (4 hours)

| Time | Activity | Type |
|------|----------|------|
| 0:00 | Day 2 Kickoff — Quick review + Codespace setup check | 10 min |
| 0:10 | **Lab 3: State & Decision-Making (Notebooks 104, 106)** | Hands-on |
| 1:10 | Break | 10 min |
| 1:20 | **Lab 4: Adding AI — Your First Smart Agent (Notebook 108)** | Hands-on |
| 2:05 | Break | 10 min |
| 2:15 | **Lab 5: AI Agents with Tools (Notebook 110)** | Hands-on |
| 3:15 | **Lab 6: Human-in-the-Loop (Notebook 111 demo)** | Demo + discussion |
| 3:40 | Closing: Where This Goes Next + Self-Study Path | 20 min |

---

## What You Need

- A laptop with a web browser (Chrome, Firefox, Edge, or Safari)
- A **GitHub account** — sign up free at [github.com](https://github.com)
- A **Claude AI account** — sign up free at [claude.ai](https://claude.ai) (or use [chat.openai.com](https://chat.openai.com))
- WiFi connection at the venue

**You do NOT need:**
- Any software installed
- Programming experience
- An API key (it's pre-configured for you)

---

# DAY 1

---

## Lab 1: YouTube → Training Curriculum Builder

**Duration:** 60 minutes
**Tools:** Claude AI (or ChatGPT) in your browser + facilitator's Open Brain system
**Goal:** Use AI to search, extract, and transform YouTube video content into a real training curriculum

---

### What you're building

You're going to build a **complete training curriculum** — study guide, lesson plan, and quiz — on a topic your group chooses. You won't write any of it manually. Instead, you'll use AI in three steps:

```
FETCH  →  STRUCTURE  →  CREATE
(search)   (summarize)   (build curriculum)
```

This is the same pattern behind almost every useful AI application in the real world.

---

### Step 1: Watch the facilitator search (10 min)

The facilitator will search for YouTube videos on a topic the group picks. Watch how the system works:

**What's happening under the hood:**
- The facilitator runs a **semantic search** — not a keyword search like Google
- Instead of matching exact words, the AI finds videos whose *meaning* matches the query
- This uses the **embeddings** and **dot products** we covered in Session 0 — the query gets converted into a vector and compared against video summary vectors in high-dimensional space
- Videos with similar meaning show up, even if they don't contain the exact search words

**Watch for:**
- How many results come back
- Whether the titles match the intent (not just the words) of the search
- What happens when the facilitator refines the query

---

### Step 2: Watch the facilitator extract summaries (10 min)

The facilitator will pull full structured summaries for 3-4 of the best videos. Watch what comes back:

**What's happening under the hood:**
- The system has already processed these videos using AI — extracting key topics, main arguments, and takeaways
- This is the **structure** step — turning a 20-minute unstructured video into organized, searchable data
- The AI used **attention** (understanding context and relationships) and **MLPs** (extracting facts) to create these summaries — the same concepts from Session 0

**What you'll see:**
- A summary of each video's main points
- Key topics and concepts extracted
- Takeaways and ratings
- This structured data is what makes the next step possible

---

### Step 3: Build your curriculum (20 min)

Now it's your turn. Open **claude.ai** (or chat.openai.com) in your browser.

The facilitator will share the video summaries with you. Copy and paste them along with the prompt below into your Claude or ChatGPT session.

#### The Prompt

Copy this entire block, replace [TOPIC] with your group's topic, and paste the video summaries where indicated:

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

[PASTE THE VIDEO SUMMARIES HERE]
```

#### What to do while it generates:

- Watch how it organizes the content — does the module sequence make sense?
- Check the study guide — are the definitions accurate?
- Look at the quiz — are the questions actually testing understanding, or just trivia?

---

### Step 4: Review and discuss (15 min)

When your curriculum is ready, review it with these questions:

1. **Did the AI get anything wrong?** Look for invented facts, misinterpretations, or claims not in the source videos.

2. **Would you actually use this to train someone?** Is it 80% there? 50%? What would you change?

3. **What took you 20 minutes that would have taken how long by hand?** Building a curriculum from scratch — with a study guide and quiz — normally takes hours or days.

4. **What else could you build with the same FETCH → STRUCTURE → CREATE pattern?**
   - Incident reports → trend analysis
   - Regulations → compliance checklist
   - Meeting notes → action tracker
   - Maintenance logs → readiness summary

**Key takeaway:** AI is not perfect, but it's an incredible accelerator. The value is in the speed — you still need a human to review, correct, and approve.

---

### Step 5: Connection to Session 0

Everything you just used is powered by the nine concepts from this morning:

| What you did | What's happening under the hood |
|---|---|
| Semantic search | **Embeddings** and **dot products** — query vector compared to video vectors |
| Summary extraction | **Attention** (context) + **MLPs** (facts) built the structured data |
| Curriculum generation | **Next-token prediction** — the model generated the curriculum word by word |
| Your review of the output | **Human-in-the-loop** — AI drafts, human approves |

---

## Lab 2: Your First AI Workflow

**Duration:** 40 minutes
**Tools:** GitHub Codespaces
**Notebooks:** 101, 102, 103
**Goal:** Understand the building blocks of AI workflows — data structures, nodes, edges, and your first graph

---

### Getting into your Codespace

If you haven't set up your Codespace yet, do it now:

1. Go to [github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop)
2. Click the green **Code** button → **Codespaces** tab → **Create codespace on main**
3. Wait 2-3 minutes. You'll see: `✅ Environment ready! API key configured.`
4. In the left sidebar, open the **notebooks** folder

---

### Notebook 101: Type Annotations (~10 min)

**Open:** `notebooks/101_type_annotations.ipynb`

**What this is about:**
Before you can build an AI workflow, you need a way to describe the data flowing through it. This notebook teaches you how to define **data structures** — templates that describe what information exists and what type it is.

**Analogy:** Think of it like a form template. Before anyone fills out the form, you define exactly which fields exist and what kind of data goes in each one. "Name" is text. "IP address" is text. "Active" is yes/no. If someone tries to put the wrong type of data in a field, the system catches it.

**How to work through it:**
1. Click on the first code cell (the gray box with code in it)
2. Press **Shift + Enter** to run it — the output appears below
3. Read the explanation text between code cells
4. Keep pressing **Shift + Enter** to move through the notebook
5. Don't worry about understanding every line of code — focus on the **concepts** in the text

**Key things to watch for:**
- **TypedDict** — the way Python defines structured data (like a form template)
- **Optional fields** — some data might not always be present
- **Why this matters:** Every AI workflow needs a state definition — this is where it starts

**When you're done:** You understand that AI workflows need structured data definitions, just like a database needs a schema or a form needs fields defined before anyone fills it out.

---

### Notebook 102: Core Concepts (~15 min)

**Open:** `notebooks/102_core_concepts.ipynb`

**What this is about:**
This notebook introduces the three building blocks of every AI workflow:
- **State** — the data that flows through the system (the clipboard traveling down an assembly line)
- **Nodes** — the workstations that do the actual work (each one reads the clipboard, does one job, writes results back)
- **Edges** — the conveyor belts connecting workstations (which station comes after which)

**Analogy:** An assembly line. The **state** is the clipboard of information traveling with the product. Each **node** is a workstation that reads the clipboard, does its job, and writes its results back. **Edges** are the conveyor belts connecting stations. The **graph** is the whole factory floor.

**Key things to watch for:**
- How state is defined (TypedDict from notebook 101)
- How nodes are just Python functions: they receive state, do something, return updates
- How edges connect nodes in sequence: START → node A → node B → END
- **Partial state updates** — a node only changes the fields it cares about, not everything

**When you're done:** You understand that an AI workflow is built from three simple pieces: data (state), processing steps (nodes), and connections (edges). That's it.

---

### Notebook 103: Your First Graph (~15 min)

**Open:** `notebooks/103_first_graph.ipynb`

**What this is about:**
Now you build your first actual workflow. This is the "hello world" — the simplest possible version. One step comes in, one workstation processes it, one result comes out.

**What you're building:**
A simple address validation workflow: you give it network configuration data, it validates it, and returns the result. Three pieces: a starting point (START), a processing step (the validate node), and a finish line (END).

**Key things to watch for:**
- `StateGraph(MyState)` — creating the factory floor
- `graph.add_node("name", function)` — placing a workstation
- `graph.add_edge(START, "name")` — connecting the conveyor belt
- `graph.compile()` — activating the factory (making it runnable)
- `app.invoke(initial_data)` — sending the first product down the line

**When you're done:** You've built and run your first AI workflow graph. Every complex system you'll see for the rest of this workshop is just more of these simple pieces snapped together.

---

### Day 1 Wrap-Up (5 min)

**What you accomplished today:**
1. You understand how AI works under the hood (Session 0)
2. You used AI to build a real training curriculum from video content (Lab 1)
3. You built your first workflow graph in code (Lab 2)

**What's coming tomorrow:**
- Making workflows smarter with decision points and complex data
- Plugging in actual AI (Claude) so the workflow can *think*
- Building an agent that can use tools and reason about problems
- Seeing how humans and AI collaborate in a loop

---

# DAY 2

---

### Day 2 Kickoff (10 min)

**Quick review — what we covered yesterday:**
- AI is a function with learned parameters — neurons, layers, weights, trained via gradient descent
- Modern AI (GPT/Claude) uses the Transformer architecture: embeddings, attention, MLPs
- The FETCH → STRUCTURE → CREATE pattern
- Workflow building blocks: state, nodes, edges, graphs

**Codespace check:**
- Everyone open your Codespace: go to [github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop) → **Code** → **Codespaces** tab → click your existing Codespace (don't create a new one)
- If yours is gone, create a new one — it takes 2-3 minutes

---

## Lab 3: State & Decision-Making

**Duration:** 60 minutes
**Notebooks:** 104, 106
**Goal:** Handle complex data and add decision points to your workflows

*Note: We're skipping notebook 105 (sequential workflows) — you already saw that pattern in 103, and 105 just extends it to more steps. It's available for self-study.*

---

### Notebook 104: State Management (~25 min)

**Open:** `notebooks/104_state_management.ipynb`

**What this is about:**
Real jobs need more than a couple of fields on the clipboard. You need to track the rule name, which network zones it applies to, whether it passed validation, any errors found, how many times you've retried, and the final result. This notebook teaches you how to manage complex state with many fields and different data types.

**Key things to watch for:**
- State with 10+ fields of different types (text, numbers, lists, yes/no)
- **Safe state access** — only read fields that have been set by a previous step
- How state accumulates as it flows through multiple nodes — each node adds its piece
- Working with lists (multiple zones, multiple tags, multiple applications)

**Analogy:** A military operations form with many sections. Each section gets filled in by a different staff officer. The S2 fills in the intel section, the S3 fills in the operations section, the S4 fills in the logistics section. Nobody reads a section that hasn't been filled in yet.

**When you're done:** You can design state schemas that track complex, multi-field data across a workflow.

---

### Notebook 106: Conditional Routing (~35 min)

**Open:** `notebooks/106_conditional_routing.ipynb`

**What this is about:**
So far, workflows have been straight lines — START → step 1 → step 2 → END. But real workflows need decision points. "If this is a production config, require approval. If it's a dev config, go straight to creation." This notebook teaches you how to add **branching** to your workflows.

**Analogy:** Think of it like a sorting facility — or a firewall ACL. A package arrives, a worker reads the label, and sends it down the right conveyor belt. The worker doesn't open the package or change anything — they just read and route. "Development config? Go left. Production config? Go right. Staging? Middle lane."

**Key things to watch for:**
- **Router functions** — they examine the state and return which path to take (they don't change anything — they just decide)
- **`add_conditional_edges()`** — the key API that enables branching
- **The lambda passthrough pattern** — `lambda state: state` — this is a quirk of LangGraph, don't overthink it, just use it
- **Multi-way routing** — you can have 2, 3, or N branches

**The critical pattern:**
```python
# 1. Create a router node (does nothing, just passes state through)
graph.add_node("router", lambda state: state)

# 2. Define a function that looks at state and decides
def which_path(state) -> Literal["path_a", "path_b"]:
    if state["requires_approval"]:
        return "path_a"
    return "path_b"

# 3. Wire it up
graph.add_conditional_edges("router", which_path, {
    "path_a": "approval_node",
    "path_b": "direct_create_node"
})
```

**When you're done:** You can build workflows that make decisions — routing data down different paths based on conditions. This is where workflows start to feel intelligent.

---

## Break (10 min)

---

## Lab 4: Adding AI — Your First Smart Agent

**Duration:** 45 minutes
**Notebook:** 108
**Goal:** Connect a real AI model (Claude) into your workflow graph

*Note: We're skipping notebooks 105, 107, and 109. They're available for self-study and cover sequential chains, loops, and conversation memory — important topics, but we're prioritizing getting you to the "aha moment" of AI integration.*

---

### Notebook 108: First LLM Integration (~45 min)

**Open:** `notebooks/108_first_llm_integration.ipynb`

**What this is about:**
This is where it gets exciting. Until now, every node in your workflows has been a regular Python function — fixed logic, predictable output. Now you're going to replace one of those nodes with **an actual AI model**. Instead of a workstation running a formula, you're putting an analyst at the workstation — one who can read a question in plain English, think about it, and write an answer.

**Connection to Session 0:**
Remember the architecture we covered yesterday? Tokens → embeddings → 96 layers of attention and MLPs → next-token prediction? That's exactly what's happening when you call `llm.invoke()`. Your text gets tokenized, embedded into 12,288-dimensional vectors, processed through attention heads and feed-forward layers, and the model predicts the most helpful response word by word.

**Key things to watch for:**

1. **`ChatAnthropic`** — this is the LangChain wrapper around Claude's API. When you call it, you're sending your text to Claude's servers, which run it through the Transformer architecture
2. **`HumanMessage`** — this is how you package your question for the AI. It's a structured object, not just a raw string
3. **The graph is the same pattern** — `StateGraph`, `add_node`, `add_edge`, `compile`, `invoke`. The only difference is that the node function calls an AI instead of running fixed code
4. **The memory problem** — this simple bot doesn't remember previous messages. Ask it a question, it answers. Ask a follow-up, it has no idea what you asked before. We'll fix this in the next lab

**What you're building:**
A simple bot that can answer questions about PAN-OS firewalls. It uses Claude Haiku (a fast, affordable version of Claude) as its brain. You ask a question, the AI processes it and responds.

**Try these queries:**
- "What is PAN-OS?"
- "How do I check the firewall version from the CLI?"
- "What's the difference between allow and drop actions in a security rule?"
- Then try asking a follow-up like "Can you tell me more about that?" — watch it fail because it has no memory of what "that" refers to

**When you're done:** You've connected a real AI into a workflow graph. The graph doesn't care whether the workstation runs a formula or asks an AI — it's all just "state in, state out." But you've also seen the limitation: without memory, the AI can't have a real conversation.

---

## Break (10 min)

---

## Lab 5: AI Agents with Tools

**Duration:** 60 minutes
**Notebook:** 110
**Goal:** Build an agent that can reason about problems and use tools to take action

---

### Notebook 110: ReAct Agents with Tools (~60 min)

**Open:** `notebooks/110_react_agents_with_tools.ipynb`

**What this is about:**
This is the capstone lab. Everything we've learned comes together here. You're going to build an **AI agent** — software that can:
1. **Reason** about a problem (using the AI model)
2. **Choose** which tool to use (using attention and learned knowledge)
3. **Execute** the tool (calling a Python function)
4. **Evaluate** the result and decide whether to use another tool or finish
5. **Repeat** until the job is done

This is the full loop. This is what people mean when they say "AI agent."

**Analogy (from Session 0):** Remember the three layers?
- **Workflow engine** (LangGraph) = the plumbing → that's the graph you're building
- **AI brain** (Claude) = the decision-maker → that's the LLM node
- **Tools** (Python functions) = the hands → that's what you're adding in this lab

**Key concepts in this notebook:**

1. **The `add_messages` reducer** — This fixes the memory problem from notebook 108. Instead of manually tracking conversation history, the reducer automatically appends every new message. Think of it like a court reporter who records everything — you don't have to tell them to write things down.

2. **Tools with `StructuredTool`** — You create Python functions and register them as tools the AI can call. The critical part: the **docstring** (description) is how the AI decides when to use the tool. The AI reads the description and decides "this tool is relevant to the question I was just asked." Bad description = AI never uses the tool.

3. **The ReAct loop** — The core pattern:
   ```
   User asks a question
     → AI reasons about which tool to use
       → Tool executes and returns result
         → AI reads the result
           → AI decides: need another tool, or ready to answer?
             → If another tool: loop back
             → If ready: respond to user
   ```

4. **`ToolNode`** — A prebuilt LangGraph component that handles tool execution. You don't have to write the tool-calling machinery yourself.

5. **Conditional routing (again!)** — The `should_continue` function checks if the AI wants to call a tool or is ready to give a final answer. If tool call → route to the tool node. If no tool call → route to END.

**The tools you'll build:**
- `check_panos_version` — Query a firewall's current software version
- `calculate_upgrade_downtime` — Estimate how long an upgrade will take
- `check_upgrade_compatibility` — Verify if an upgrade path is valid

**Try these queries once it's built:**
- "What version is fw-prod-01 running?" (uses one tool)
- "Check fw-prod-01's version, then tell me how long it would take to upgrade to 11.0.0" (uses two tools in sequence — the AI decides on its own to chain them)
- "Can fw-prod-02 upgrade to 11.0.0? If so, how long will it take?" (uses three tools — compatibility check, version check, downtime estimate)

**Watch the output carefully:**
- See the AI "thinking" about which tool to use
- See the tool execute and return structured data
- See the AI read the result and decide what to do next
- Notice how the AI chains tools together without you telling it the order — it figures out the dependency itself

**Connection to Session 0:**
When the AI "decides" which tool to use, here's what's happening inside the model:
- The conversation (including the tool descriptions) is tokenized and embedded
- **Attention** weighs which parts of the context matter — "the user asked about versions, and there's a tool called check_panos_version with a description that matches"
- **MLPs** inject knowledge about when version checks are relevant
- **Next-token prediction** generates a tool call as the most likely next output

It's not magic. It's the same architecture from Session 0, applied to tool selection.

**When you're done:** You've built a genuine AI agent — one that can reason, choose tools, execute them, and loop until the job is done. This is the same pattern powering enterprise AI assistants, customer service bots, and automated operations platforms.

---

## Lab 6: Human-in-the-Loop (Facilitator Demo)

**Duration:** 25 minutes
**Notebook:** 111
**Format:** Facilitator-led demo with group discussion — students watch, not hands-on
**Goal:** See how humans and AI collaborate in an approval workflow

---

### Why this is a demo, not hands-on

Notebook 111 involves interactive input (typing into the running notebook) and is the most complex pattern in the workshop. Rather than have 30 people debug interactive prompts, the facilitator will run this live while the group watches and discusses.

---

### What to show

**Setup explanation (3 min):**

> "We've built an agent that can reason, use tools, and loop. But there's one missing piece for real-world use: human approval. You probably don't want an AI making production changes without someone signing off. That's what Human-in-the-Loop means."

**Live demo (15 min):**

Run notebook 111 and walk through the conversation with the AI:

1. **"Create a NAT policy for our web servers"**
   - The AI drafts a configuration using the `update_configuration` tool
   - Show the output — a structured NAT policy

2. **"Change the source zone to 'dmz' instead of 'trust'"**
   - The AI modifies the draft and updates using the same tool
   - Point out: the AI kept the rest of the config and only changed what you asked

3. **"Add a description: 'Web server outbound NAT'"**
   - Another refinement. The AI is iterating based on human feedback.

4. **"Looks good, save it as web-nat-policy"**
   - The AI calls `save_configuration` instead of `update_configuration`
   - The router function detects the save tool was used → routes to END
   - The loop terminates. The workflow is complete.

**Key points to make during the demo:**

> "Notice what's happening architecturally. Two tools: update (loops back to the agent for more feedback) and save (routes to END). The conditional routing we learned in notebook 106 is what controls this. The AI doesn't decide when to stop — the *graph* decides, based on which tool was called. The human decides when the work is good enough."

> "This is the pattern behind every 'copilot' product you've heard about — GitHub Copilot, Microsoft 365 Copilot, etc. The AI drafts, the human reviews and refines, and the human decides when to approve and commit."

**Discussion (7 min):**

1. **"Where would you want this pattern in your work?"**
   - Operations orders — AI drafts, human reviews
   - Maintenance work orders — AI generates from sensor data, supervisor approves
   - Incident reports — AI structures raw notes, analyst verifies and signs
   - Training schedules — AI proposes, S3 adjusts and approves

2. **"What's the risk of removing the human from this loop?"**
   - AI can hallucinate — it might generate a valid-looking config that's wrong
   - Accountability — someone needs to own the decision
   - Context the AI doesn't have — political considerations, morale, timing
   - Trust must be earned incrementally — start with human-in-the-loop, expand autonomy as confidence grows

---

## Closing: Where This Goes Next (20 min)

---

### What you built over two days

| Day | Lab | What you did |
|-----|-----|-------------|
| 1 | Session 0 | Understood how AI works: neurons → layers → gradient descent → transformers → attention |
| 1 | Lab 1 | Used AI to build a training curriculum from YouTube content (FETCH → STRUCTURE → CREATE) |
| 1 | Lab 2 | Built your first workflow graph in code (state, nodes, edges) |
| 2 | Lab 3 | Added complex data and decision-making (conditional routing) |
| 2 | Lab 4 | Connected a real AI model into your workflow |
| 2 | Lab 5 | Built an agent that reasons, uses tools, and loops |
| 2 | Lab 6 | Saw human-AI collaboration in an approval workflow |

### The six patterns

Every AI agent system you'll encounter uses some combination of these six patterns:

| # | Pattern | Notebook | Plain English |
|---|---------|----------|---------------|
| 1 | State management | 103-104 | Define what data flows through the system |
| 2 | Sequential processing | 105 | Do steps in order: A → B → C |
| 3 | Conditional routing | 106 | Make decisions: if X, go left; if Y, go right |
| 4 | Looping | 107 | Repeat until a condition is met |
| 5 | AI integration + tools | 108, 110 | Let AI reason and use tools to act |
| 6 | Human-in-the-loop | 111 | AI drafts, human approves |

These patterns are domain-agnostic. The network security examples are just one lens. The same six patterns apply to logistics, intelligence, maintenance, cybersecurity, HR, finance — any workflow.

### Self-study path

These notebooks were skipped in the workshop but are available in your Codespace:

| Notebook | Topic | Why it matters |
|----------|-------|----------------|
| 105 | Sequential workflows | Multi-step pipelines with dependencies |
| 107 | Looping workflows | Retry logic, pagination, polling |
| 109 | Conversational memory | How AI remembers context across messages |

Your Codespace will stay available after the workshop (GitHub free accounts include 60 hours/month). To continue on your own after the workshop API key expires, you'll need your own Anthropic API key — see the Student Quickstart Guide for instructions.

### Resources

- **Workshop repo:** [github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop)
- **3Blue1Brown Deep Learning series:** [youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **Claude AI:** [claude.ai](https://claude.ai)
- **Anthropic API (for building your own apps):** [console.anthropic.com](https://console.anthropic.com)
- **LangGraph documentation:** [python.langchain.com/docs/langgraph](https://python.langchain.com/docs/langgraph)

---

## Facilitator Notes

### Pacing guidance

The tightest segments are Labs 4 and 5 on Day 2. If you're running behind:
- **Lab 4 (Notebook 108):** Can compress to 30 min by having students just run cells and observe, rather than experimenting with their own queries
- **Lab 5 (Notebook 110):** The first half (building the tools) is the most important. If time is short, run the multi-tool queries as a facilitator demo instead of individual exploration
- **Lab 6 (Notebook 111):** Already a demo — can compress to 15 min by cutting the discussion short

### Common issues on Day 2

| Issue | Fix |
|-------|-----|
| Codespace timed out overnight | Students click their existing Codespace to restart it (~30 sec). If deleted, create a new one (~2 min). |
| "API key not found" error | The `.env` file should be auto-created. If missing: have student run `echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" > .env` in the terminal |
| Notebook kernel dies | Click Restart in the kernel menu. Re-run cells from the top. |
| Rate limiting on Claude | Free-tier Anthropic accounts have rate limits. Wait 60 seconds between notebook cells if hitting limits. With the shared workshop key, this shouldn't be an issue unless all 30 students run simultaneously. |
| Student is lost in code | Remind them: "Focus on the concepts in the text between cells, not the syntax. The code is there so you can see it's real — you don't need to understand every line." |
| Student finishes early | Point them to the self-study notebooks (105, 107, 109) or suggest they experiment with different queries in 108/110 |

### Bridging language

Use these phrases throughout Day 2 to connect labs back to Session 0:

- "Remember embeddings? That's what's happening when the AI reads your question — it gets tokenized and converted into vectors."
- "The AI is using attention right now — weighing which parts of the conversation and tool descriptions are most relevant to your question."
- "When the AI chooses a tool, that decision is happening in the MLP layers — the same fact-storage mechanism we talked about yesterday."
- "Next-token prediction is generating this response right now — one word at a time, left to right."
- "This is gradient descent in action — 300 billion training examples tuned 175 billion parameters so the model would make good tool-calling decisions in situations like this."
