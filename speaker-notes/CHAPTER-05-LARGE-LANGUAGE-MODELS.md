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

- Grant opens this video with a great analogy — imagine a movie script
  - A character is interacting with an AI on screen
  - Someone tears off the AI's response — your job is to guess what it said
  - To predict well, you'd draw on everything you know about how AI assistants typically respond
- That's exactly what a large language model does
  - It takes all the text so far and predicts the most likely continuation
  - Not by "understanding" in the human sense — by recognizing patterns from training data
- This framing is useful because it strips away the mystique
  - No consciousness, no understanding, no intent
  - Just: "given this setup, what text would a human most likely have written next?"

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

- Let's be precise about what an LLM is
  - It is a mathematical function — numbers in, numbers out
  - The input: your text, converted to token IDs (we covered tokenization earlier)
  - The output: a probability for every word in its vocabulary
- It doesn't pick just one answer — it scores all ~50,000 possible next tokens
  - "Paris" might get 68%, "a" might get 4%, "the" might get 3%
  - Then it samples from that distribution — usually picking a high-probability word, but with some randomness
- Then it appends the chosen word to the input and runs the whole function again
  - "The capital of France is Paris" → now predict the next word after "Paris"
  - This is how it generates entire paragraphs — one word at a time, autoregressively
- Important: it doesn't look up answers in a database or search the internet (unless given tools to do so)
  - Everything comes from the patterns encoded in its parameters during training

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

- Here's the trick that turns a text predictor into a chatbot
  - You lay out a specific format: system message, then user message, then "Assistant:"
  - The LLM sees this formatted text and predicts what would come after "Assistant:"
  - It doesn't "know" it's an assistant — it's just completing the pattern
- The system message is like stage directions for an actor
  - "You are a helpful assistant" → the model predicts helpful-sounding text
  - "You are a pirate" → the model predicts pirate-sounding text
  - Same model, same parameters — just different text context leading to different predictions
- This is why ChatGPT, Claude, and other chatbots all feel similar despite being different models
  - They're all LLMs trained on similar data, given similar conversation formats
  - The differences come from training data choices, fine-tuning approaches, and system prompts

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

- The model's entire "intelligence" lives in its parameters
  - Weights in the attention layers (we covered query, key, value matrices)
  - Weights in the MLP/feed-forward layers (the knowledge storage layers)
  - Biases at each neuron
  - Embedding values for each token
- GPT-3 has 175 billion of these numbers
  - That's roughly one parameter for every star in the Milky Way galaxy
  - GPT-4 is rumored to be much larger — exact number not publicly confirmed
  - Claude's parameter count is also not public, but it's in the same ballpark
- With random parameters, the model outputs pure gibberish
  - Not even grammatically correct — just random token sequences
  - The magic is entirely in how those parameters get set during training

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

- Training is exactly the process we covered in earlier chapters — just at enormous scale
  - Same gradient descent, same backpropagation, same cost function
  - The cost: "how surprised was the model by the actual next word?"
  - If the model predicted "Paris" with high probability and the answer was "Paris" — low cost, small adjustment
  - If the model predicted "banana" with high probability and the answer was "Paris" — high cost, big adjustment
- The training data is a massive corpus of text from the internet
  - Books, websites, Wikipedia, code, scientific papers, forums, news articles
  - GPT-3 trained on roughly 300 billion tokens of text
  - The model sees each example, makes a prediction, gets corrected, and adjusts — billions of times
- The progression from gibberish to coherence is gradual
  - First the model learns basic character and word patterns
  - Then grammar and sentence structure
  - Then facts and relationships
  - Then nuance, tone, and reasoning patterns
  - Each step is a tiny nudge — but billions of tiny nudges add up to something remarkable

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

- Grant puts the computational scale in perspective with a striking number
  - If you could do a billion operations per second — which is roughly what a modern CPU does
  - Training GPT-3 would take over 100 million years
  - That's longer than the time since dinosaurs went extinct
- The solution: massive parallelism
  - Thousands of GPUs (graphics processing units) or TPUs (tensor processing units) working simultaneously
  - Training runs last weeks to months even with this hardware
  - The electricity bill alone is staggering
- Important distinction: training vs. inference
  - Training: the expensive, one-time process of setting the parameters (months, millions of dollars)
  - Inference: using the trained model to generate text (milliseconds, fractions of a cent per query)
  - When you use ChatGPT, the model is already trained — you're just running inference
  - The parameters are frozen — they don't change when you chat with it

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

- Most people don't realize there are two distinct training phases
- **Pre-training** is what we've been discussing: predict the next word on billions of text examples
  - This produces a model that's very good at continuing text
  - But it's not a helpful assistant — it might continue a question with *another* question
  - Or it might produce toxic content if the training data contained toxic text
  - It's a mirror of the internet — brilliant and terrible in equal measure
- **RLHF** — Reinforcement Learning from Human Feedback — is the second phase
  - Humans generate pairs of responses and pick the better one
  - "Response A is helpful and accurate. Response B is evasive and wrong."
  - The model's parameters get adjusted to produce more responses like A and fewer like B
  - This is what turns a raw text predictor into a polite, helpful, safety-conscious assistant
- This two-phase process is why the models are called "pre-trained" (the P in GPT)
  - Phase 1 gives it knowledge and language ability
  - Phase 2 gives it manners and helpfulness
- The RLHF phase uses far less compute than pre-training but has an outsized effect on behavior

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

- We covered attention in depth earlier, so this is a quick recap in the LLM context
- Before the Transformer architecture (introduced in 2017), language models processed text sequentially
  - Read word 1, update internal state, read word 2, update, read word 3...
  - Like reading a book one word at a time with a tiny notepad for memory
  - This was slow and made it hard to connect distant words
- The Transformer changed everything by processing all words at once
  - Every word can "look at" every other word simultaneously through attention
  - This is massively parallelizable — GPUs can compute thousands of attention scores at the same time
  - Grant emphasizes this: a big reason Transformers won is that they're *efficient to run on modern hardware*
- Two alternating layer types:
  - Attention: words talk to each other, update meanings based on context
  - Feed-forward/MLP: each word processed independently, factual knowledge injected
- The combination of parallel processing + attention + scale is what produced the breakthrough
  - Same underlying math as our simple digit network — just organized differently and scaled enormously

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

- This is the point Grant ends on, and it's worth sitting with
  - Nobody at OpenAI or Anthropic programmed "translate French to English"
  - Nobody coded "write a sonnet in iambic pentameter"
  - Nobody specified "when asked about chemistry, use correct formulas"
  - All of these abilities emerged from one training objective: predict the next word
- The parameters were set by gradient descent — billions of tiny adjustments over billions of examples
  - No human decided what value weight #47,382,019,445 should have
  - The training process found values that reduce prediction error — and complex behaviors emerged as a byproduct
- This creates a fundamental interpretability challenge
  - We can't open up the model and point to "this is the part that knows French"
  - The knowledge is distributed across billions of parameters in ways we're still learning to understand
  - Researchers are actively working on interpretability — figuring out what individual neurons and circuits represent
- Practical implication for this audience:
  - LLMs are powerful tools, but they're not fully predictable
  - They can produce impressive results and also make confident-sounding mistakes
  - Understanding *what* they are (a next-word predictor trained on internet text) helps you use them wisely
  - They don't have beliefs, intentions, or understanding — they have patterns
- This connects directly to why we're building agents in the workshop
  - The LLM provides the language ability — pattern matching, text generation, reasoning-like behavior
  - But we add structure around it — tools, workflows, guardrails — to make it reliable and useful
  - Understanding the foundation helps you know when to trust the AI and when to verify

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
