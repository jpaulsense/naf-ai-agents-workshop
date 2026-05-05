# Student Quickstart Guide

## Before the Workshop (5 minutes)

Please complete these two steps before Day 1. Both are free and take about 2 minutes each.

### 1. Create a GitHub account

- Go to [github.com](https://github.com) and click **Sign up**
- If you already have an account, you're set — just make sure you can log in

### 2. Create a Claude AI account

- Go to [claude.ai](https://claude.ai) and click **Sign up**
- You can sign up with your Google account or an email address
- Free accounts work fine for this workshop
- If you prefer ChatGPT, that works too — go to [chat.openai.com](https://chat.openai.com)

That's it. **No software to install.** Everything runs in your browser.

---

## Day 1 — Lab 1: YouTube Curriculum Builder

This is a self-paced lab. Open `LAB-1-YOUTUBE-CURRICULUM-BUILDER.md` and follow the steps at your own pace.

**What you'll need open:**
- One browser tab with [claude.ai](https://claude.ai) (or [chat.openai.com](https://chat.openai.com))
- The `transcripts/` folder in this repository (your source material)
- That's it

---

## Day 1 & Day 2 — Codespace Lab: Building AI Agents

### Step 1: Open the workshop repository

Go to: **[github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop)**

### Step 2: Launch your Codespace

1. Click the green **Code** button
2. Click the **Codespaces** tab
3. Click **Create codespace on main**

![Create Codespace](https://github.com/codespaces/badge.svg)

### Step 3: Wait for setup (~2-3 minutes)

Your cloud environment is building. You'll see a terminal with progress messages. When it's done, you'll see:

```
✅ Environment ready! API key configured.
```

**The API key is already set up for you — no configuration needed.**

### Step 4: Open the first notebook

1. In the file browser on the left side, click the **notebooks** folder
2. Open **101_type_annotations.ipynb**
3. If asked to select a kernel, choose **.venv (Python 3.11)**

### Step 5: Run your first cell

- Click on the first code cell
- Press **Shift + Enter** to run it (or click the ▶ play button)
- The output appears below the cell
- Keep pressing **Shift + Enter** to move through the notebook

---

## Quick Reference

### Keyboard shortcuts in Jupyter

| Action | Shortcut |
|--------|----------|
| Run a cell and move to next | **Shift + Enter** |
| Run a cell and stay | **Ctrl + Enter** |
| Add a cell below | **B** (when not editing a cell) |
| Delete a cell | **D, D** (press D twice, when not editing) |
| Undo delete | **Z** (when not editing) |
| Switch to edit mode | **Enter** |
| Switch to command mode | **Escape** |

### Notebook order

Follow the notebooks in number order. The facilitator will guide you through which ones to work on during each session.

**Phase 1 — No AI needed (these use practice data):**
| Notebook | Topic |
|----------|-------|
| 101 | Type annotations — defining data structures |
| 102 | Core concepts — nodes, edges, state |
| 103 | Your first graph — the simplest workflow |
| 104 | State management — tracking complex data |
| 105 | Sequential workflows — multi-step pipelines |
| 106 | Conditional routing — decision points |
| 107 | Looping workflows — repeat until done |

**Phase 2 — Uses Claude AI (API key is pre-configured):**
| Notebook | Topic |
|----------|-------|
| 108 | First LLM integration — connecting AI |
| 109 | Conversational memory — AI that remembers |
| 110 | ReAct agents with tools — AI that acts |
| 111 | Human-in-the-loop — AI that asks permission |

### Troubleshooting

**"Kernel not found" or "Module not found":**
1. Press **F1** (or **Cmd+Shift+P** on Mac)
2. Type: `Codespaces: Rebuild Container`
3. Select it and wait for the rebuild (~2 minutes)

**Cell seems stuck / not responding:**
- Click **Restart** in the kernel menu at the top
- Re-run the cells from the beginning of the notebook

**Accidentally closed the browser tab:**
- Go back to [github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop)
- Click **Code** → **Codespaces** tab
- Your existing Codespace will be listed — click it to reopen (don't create a new one)

**Need help:**
- Raise your hand — the facilitator is here to help
- Pair up with a neighbor if your environment isn't cooperating

---

## After the Workshop

### Keep learning on your own

Your Codespace will stay available for a while after the workshop. GitHub free accounts include 60 hours of Codespace time per month.

To continue on your own after the workshop, you'll need your own Anthropic API key for notebooks 108-111:

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account and add a payment method
3. Create an API key
4. In your Codespace terminal, run: `echo "ANTHROPIC_API_KEY=your-key-here" > .env`

**Cost:** Notebooks 108-111 use Claude Haiku, which costs about $0.25 per million tokens. A full run through all four notebooks costs roughly $0.50-2.00.

### Fork the repo

If you want your own permanent copy:
1. Go to [github.com/jpaulsense/naf-ai-agents-workshop](https://github.com/jpaulsense/naf-ai-agents-workshop)
2. Click **Fork** (top right)
3. Now you have your own copy that won't go away
