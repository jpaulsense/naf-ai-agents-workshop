# Chapter 5 — Large Language Models Explained Briefly

## Presentation Guide

**Duration:** 25-30 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Give the audience a concise, intuitive understanding of what large language models are, how they're built, and why they behave the way they do — bridging the neural network foundations from earlier chapters to the practical AI tools they'll use in the workshop.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 5: Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **The movie script analogy** | 3 min |
| 0:03 | **What an LLM actually is** | 3 min |
| 0:06 | **How chatbots are built from LLMs** | 3 min |
| 0:09 | **Parameters: billions of tunable dials** | 3 min |
| 0:12 | **Training: from gibberish to genius** | 4 min |
| 0:16 | **The staggering scale of computation** | 3 min |
| 0:19 | **Pre-training vs. RLHF** | 3 min |
| 0:22 | **Transformers and attention (recap)** | 3 min |
| 0:25 | **Emergent behavior — why we can't fully explain it** | 3 min |

**Total: ~28 minutes**

---

## The Movie Script Analogy (3 min)

### SLIDE 01 — Imagine a movie script

**Headline:** Picture a movie script where someone asks an AI a question — and the AI's response has been torn off.

**Body:**
- Imagine a scene in a movie: a character types a question into a computer
- The AI's response appears on screen — but someone has ripped that part of the script away
- Your job is to predict what the screenwriter wrote as the AI's answer
- That's essentially what an LLM does — it predicts the most likely next words, given everything that came before

**Visual concept:** A torn movie script page. The top half shows dialogue: "USER: What is the capital of France?" The bottom half — where the AI's response would be — is ripped away, with a jagged edge. A magnifying glass hovers over the torn edge with the question: "What words come next?"

**Analogy card:**
An LLM is a text-completion machine. It doesn't "know" things the way you do. It predicts what text would most plausibly come next, based on patterns learned from billions of pages of human writing.

**Speaker notes:**

- Movie script analogy: character asks AI a question, AI's response is torn off — your job is to predict what was written
- LLM = takes all text so far, predicts most likely continuation based on patterns from training data
- No consciousness, no understanding, no intent — just "what text would plausibly come next?"

---

## What an LLM Actually Is (3 min)

### SLIDE 02 — A mathematical function that predicts the next word

**Headline:** An LLM is a function: text in, probability distribution out.

**Body:**
- Input: a sequence of text (tokens)
- Output: a probability for every possible next word (~50,000+ options)
- "The capital of France is ___" → "Paris" (high probability), "Berlin" (low), "pizza" (near zero)
- The function assigns probabilities to the *entire vocabulary* — every single word gets a score
- It picks one (with some randomness), appends it, and repeats

**Visual concept:** A box labeled "LLM" with an arrow going in (text: "The weather today is") and an arrow coming out (a ranked list: "sunny 23%, going 8%, expected 6%, cold 5%, ... pizza 0.001%"). The word "sunny" is highlighted as the selected next token.

**Key insight card:**
The LLM doesn't retrieve answers from a database. It *generates* them by repeatedly asking "what word is most likely next?" — one token at a time, left to right, until it decides to stop.

**Speaker notes:**

- LLM is a mathematical function: text (as token IDs) in, probability distribution over ~50,000 possible next tokens out
- Scores entire vocabulary — "Paris" 68%, "a" 4%, "the" 3% — then samples from distribution with some randomness
- Appends chosen word, reruns the function — generates paragraphs one token at a time (autoregressive)
- Does not look up answers in a database — everything comes from patterns encoded in parameters during training

---

## How Chatbots Are Built (3 min)

### SLIDE 03 — From text predictor to conversational assistant

**Headline:** A chatbot is an LLM with a specific text format laid out in advance.

**Body:**
- Start with a template: "System: You are a helpful assistant. User: [question]. Assistant: ___"
- The LLM doesn't know it's in a conversation — it just sees text and predicts the next word
- The "assistant" role is just a formatting trick — the model predicts what text would follow that label
- Different system prompts → different "personalities" — all from the same underlying model

**Visual concept:** A formatted text block showing:
```
System: You are a helpful, harmless AI assistant.
User: What should I pack for a camping trip?
Assistant: █
```
The blinking cursor shows where the LLM starts generating. Arrows point to each section with labels: "Sets the tone," "The question," "LLM generates from here."

**Speaker notes:**

- Chatbot = LLM with a formatted template: system message → user message → "Assistant:" — model predicts what follows
- System message is stage directions: "helpful assistant" → helpful text; "pirate" → pirate text — same model, different context
- ChatGPT/Claude/etc. feel similar because they're all LLMs with similar training data and conversation formats
- Differences come from training data choices, fine-tuning, and system prompts

---

## Parameters: Billions of Tunable Dials (3 min)

### SLIDE 04 — The knobs that define behavior

**Headline:** An LLM's behavior is entirely determined by its parameters — hundreds of billions of tunable numbers.

**Body:**
- GPT-3: 175 billion parameters (weights and biases)
- Each parameter is a single number — a "dial" that was tuned during training
- Random parameters → complete gibberish
- Well-tuned parameters → coherent, knowledgeable text generation
- The parameters *are* the model — change them, and the behavior changes

**Visual concept:** A wall of tiny dials stretching to the horizon, like a massive mixing board in a recording studio. A few dials are zoomed in, showing values like 0.0037, -1.204, 0.892. Caption: "175 billion of these. Each one matters."

**Analogy card:**
Think of a massive mixing board in a recording studio — thousands of sliders and knobs. Each one on its own does almost nothing. But set them all *just right*, and you get a perfect recording. Set them randomly, and you get noise. The parameters of an LLM work the same way.

**Speaker notes:**

- All "intelligence" lives in parameters: attention weights, MLP weights, biases, embedding values
- GPT-3 = 175 billion parameters — roughly one per star in the Milky Way; GPT-4 and Claude are larger (exact counts not public)
- Random parameters = pure gibberish; the magic is entirely in how training sets those values

---

## Training: From Gibberish to Genius (4 min)

### SLIDE 05 — How the parameters get set

**Headline:** Training = showing the model billions of text examples and repeatedly adjusting parameters to make better predictions.

**Body:**
1. Start with random parameters → model outputs gibberish
2. Show it a chunk of real text — "The Eiffel Tower is located in ___"
3. The model predicts the next word (badly, at first)
4. Compare prediction to actual next word → compute the cost (how wrong it was)
5. Use **backpropagation** to figure out which parameters to adjust and by how much
6. Nudge parameters slightly in the right direction → model gets a tiny bit better
7. Repeat billions of times across terabytes of text

**Visual concept:** A timeline showing the model's output quality improving:
- Step 1: "sdkjf the and and pizza 🗑️" (random gibberish)
- Step 1,000: "The dog was in house the" (broken grammar)
- Step 1,000,000: "The dog was sitting in the house quietly" (decent)
- Step 1,000,000,000: "The Eiffel Tower, completed in 1889, stands 330 meters tall..." (knowledgeable)

**Speaker notes:**

- Same gradient descent / backpropagation from earlier chapters, just at enormous scale
- Cost function = "how surprised was the model by the actual next word?" — high confidence + wrong answer = big adjustment
- Training data: books, Wikipedia, code, papers, forums, news — GPT-3 trained on ~300 billion tokens
- Progression: character patterns → grammar → facts → nuance/reasoning — billions of tiny nudges compound

---

## The Staggering Scale of Computation (3 min)

### SLIDE 06 — Numbers that break your intuition

**Headline:** Training an LLM requires computation on a scale that's hard to comprehend.

**Body:**
- Grant's calculation: if a single computer performed a billion operations per second, training would take **over 100 million years**
- That's why thousands of specialized chips (GPUs/TPUs) work in parallel for weeks or months
- Estimated training cost for GPT-3: $4-12 million in compute alone
- GPT-4 estimated at $100+ million
- Once trained, the model is "frozen" — using it (inference) is much cheaper than training it

**Visual concept:** A timeline comparison showing "100 million years" stretched across the slide, with markers for context: "Dinosaurs went extinct 66 million years ago." Below it: "Or: thousands of GPUs working together for a few months." A dollar sign icon showing $4-12M for GPT-3 and $100M+ for GPT-4.

**Key insight card:**
The training is the expensive part — it happens once. After that, the model is frozen and can be used (inference) millions of times at relatively low cost. You're not paying to train the model when you use ChatGPT — you're paying for inference.

**Speaker notes:**

- At 1 billion ops/sec, training GPT-3 would take 100+ million years — longer than since dinosaurs went extinct
- Solution: thousands of GPUs/TPUs in parallel, training runs last weeks-to-months; GPT-3 cost $4-12M, GPT-4 ~$100M+
- Training (expensive, one-time, months) vs. inference (cheap, milliseconds per query, parameters frozen)

---

## Pre-training vs. RLHF (3 min)

### SLIDE 07 — Two phases of training

**Headline:** First, learn to predict text (pre-training). Then, learn to be a good assistant (RLHF).

**Body:**
- **Phase 1 — Pre-training:** Auto-complete on the entire internet. The model learns language, facts, reasoning, and patterns. This produces a text predictor, not a helpful assistant.
- **Phase 2 — RLHF (Reinforcement Learning from Human Feedback):** Humans rate the model's responses. "This answer was helpful. This one was harmful. This one was evasive." The model's parameters get nudged toward producing responses humans prefer.
- Pre-training = learning *about the world*
- RLHF = learning *how to be helpful, harmless, and honest*

**Visual concept:** Two panels. Left panel (Pre-training): a funnel with "Books, Wikipedia, Code, Web pages" pouring in at the top and "Text predictor" coming out the bottom. Right panel (RLHF): a human with a thumbs-up and thumbs-down, rating two different AI responses. Arrow from the human's ratings back into the model with the label "Adjust parameters toward preferred responses."

**Analogy card:**
Pre-training is like going to school — you absorb a massive amount of general knowledge. RLHF is like on-the-job training — you learn specifically how to apply that knowledge in a way that's actually useful, professional, and appropriate.

**Speaker notes:**

- Phase 1 (pre-training): predict next word on billions of examples — produces a text predictor, not a helpful assistant
- Raw pre-trained model mirrors the internet (brilliant and terrible) — might answer a question with another question or produce toxic text
- Phase 2 (RLHF): humans rate response pairs, parameters nudge toward preferred answers — turns predictor into helpful assistant
- Pre-training = knowledge and language ability; RLHF = manners and helpfulness
- RLHF uses far less compute than pre-training but has outsized effect on behavior

---

## Transformers and Attention (3 min)

### SLIDE 08 — The architecture that made it all work

**Headline:** Transformers process all words simultaneously — not one at a time.

**Body:**
- Before Transformers (2017): models read text sequentially — one word at a time, left to right
- The breakthrough: **process all words in parallel**, letting each word attend to every other word at once
- **Attention layers:** Each word asks "which other words are relevant to my meaning?" and updates itself based on context (we covered this in detail in earlier chapters)
- **Feed-forward (MLP) layers:** Process each word independently — inject factual knowledge and refine understanding
- These two layer types alternate, repeated dozens of times (96 layers in GPT-3)

**Visual concept:** Two contrasting diagrams. Left: "Before Transformers" — words entering a funnel one at a time, like a single-lane road. Right: "Transformers" — all words entering simultaneously on a multi-lane highway, with bidirectional arrows between them labeled "attention."

**Key insight card:**
The key innovation wasn't just *what* attention does — it's that attention is massively parallelizable on GPUs. This is what enabled scaling from millions to billions of parameters. The architecture was designed for the hardware.

**Speaker notes:**

- Pre-2017 models processed text sequentially (one word at a time) — slow, hard to connect distant words
- Transformers process all words at once; every word attends to every other simultaneously — massively parallelizable on GPUs
- Two alternating layer types: attention (words talk to each other) and MLP (factual knowledge injected per word)
- GPT-3 repeats attention+MLP 96 times — same math as simple digit network, just scaled enormously
- Parallel processing + attention + scale = the breakthrough

---

## Emergent Behavior (3 min)

### SLIDE 09 — We built it, but we can't fully explain it

**Headline:** Specific capabilities emerge from training — they weren't programmed in, and they're hard to predict.

**Body:**
- Nobody programmed "be able to write poetry" or "solve logic puzzles" or "translate French to English"
- These abilities **emerged** from the training process — from optimizing one simple objective (predict the next word)
- The parameters were tuned by gradient descent, not by hand — so there's no human-readable explanation of *why* specific behaviors exist
- This is both the power and the challenge of modern AI: impressive capabilities, limited interpretability

**Visual concept:** A Venn diagram or layered circle. Center: "Predict the next word" (the training objective). Surrounding rings showing emergent capabilities: "Grammar," "Facts," "Reasoning," "Translation," "Code generation," "Humor," "Common sense." Caption: "All of these emerged from one simple goal."

**Key insight card:**
The behavior of an LLM is *emergent* — it arises from the interaction of billions of parameters, none of which were individually set by a human. This makes LLMs powerful but also fundamentally difficult to fully understand or predict.

**Speaker notes:**

- Nobody programmed "translate French" or "write a sonnet" — all emerged from one objective: predict the next word
- Parameters set by gradient descent, not humans — no one decided what weight #47B should be; complex behaviors emerged as a byproduct
- Interpretability challenge: can't point to "the part that knows French" — knowledge distributed across billions of parameters
- LLMs are powerful but not fully predictable — they produce impressive results AND confident-sounding mistakes
- No beliefs, intentions, or understanding — just patterns; understanding this helps you know when to trust vs. verify
- Workshop connection: LLM provides language ability, we add tools/workflows/guardrails around it for reliability

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| Movie script analogy | 1 (slide 01) | 3 min |
| What an LLM is | 1 (slide 02) | 3 min |
| Chatbot construction | 1 (slide 03) | 3 min |
| Parameters | 1 (slide 04) | 3 min |
| Training | 1 (slide 05) | 4 min |
| Scale of computation | 1 (slide 06) | 3 min |
| Pre-training vs. RLHF | 1 (slide 07) | 3 min |
| Transformers and attention | 1 (slide 08) | 3 min |
| Emergent behavior | 1 (slide 09) | 3 min |
| **Total** | **9 slides** | **~28 min** |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 5: "Large Language Models explained briefly" (2024) — [https://www.youtube.com/watch?v=LPZh9BOjkQs](https://www.youtube.com/watch?v=LPZh9BOjkQs)
