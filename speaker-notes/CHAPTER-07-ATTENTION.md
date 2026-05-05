# Chapter 7 — Attention in Transformers, Step by Step

## Presentation Guide

**Duration:** 35-40 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build genuine intuition for the attention mechanism — the core innovation behind Transformers — covering queries, keys, values, attention patterns, masking, multi-headed attention, and why attention enabled the AI breakthrough.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 6: "Attention in transformers, step-by-step"](https://www.youtube.com/watch?v=eMlx5fFNoYc) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **Block 1: The problem attention solves** | 8 min |
| 0:08 | **Block 2: Queries, keys, and the attention pattern** | 12 min |
| 0:20 | **Block 3: Values — the actual information exchange** | 8 min |
| 0:28 | **Block 4: Multi-headed attention and the full picture** | 10 min |
| 0:38 | **Wrap-up** | 2 min |

**Total: ~40 minutes**

---

## Block 1: The Problem Attention Solves (8 min)

### SLIDE 01 — Same word, different meaning

**Headline:** The word "mole" means completely different things depending on context.

**Visual concept:** Three panels, each showing the word "mole" in a different sentence with a different illustration:
1. "American shrew **mole**" → picture of a small burrowing animal
2. "One **mole** of carbon dioxide" → chemistry beaker with "6.022 × 10²³" label
3. "Take a biopsy of the **mole**" → dermatology diagram of a skin lesion

Below: a single embedding vector labeled "mole" with a question mark — "Same initial numbers for all three meanings?"

**Body:**
- When "mole" first enters the network, it gets the same embedding vector regardless of context
- But the meaning is completely different in each sentence
- The network needs a mechanism to update each word's meaning based on surrounding words
- That mechanism is **attention**

**Speaker notes:**

- Initial embeddings are context-free — "mole" gets same 12,288 numbers regardless of meaning
- Three meanings (animal, chemistry 6.022x10^23, skin growth) but identical starting vector
- Attention = the mechanism that lets surrounding words update each other's embeddings based on context

---

### SLIDE 02 — Context changes everything

**Headline:** Surrounding words don't just tweak meaning — they can completely transform it.

**Visual concept:** Two pairs of examples:
1. "Eiffel **tower**" → image of the massive iron Eiffel Tower in Paris vs. "miniature **tower**" → tiny decorative tower
2. "**Harry**" after "wizard" → Harry Potter illustration vs. "**Harry**" after "Queen" + "Sussex" → Prince Harry photo

**Body:**
- "Tower" starts as a generic structure — "Eiffel" transforms it into wrought iron, Paris, 1,000 feet tall
- "Harry" after "wizard" → Harry Potter. "Harry" after "Queen" and "Sussex" → Prince Harry.
- The initial embedding is a blank canvas — attention paints the specific meaning
- Every word in the context contributes information that shapes the final representation

**Speaker notes:**

- "Tower" starts identical in "Eiffel tower" and "miniature tower" — "Eiffel" injects Paris/iron/1000ft; "miniature" injects small/decorative
- "Harry" is ambiguous — "wizard" context → Harry Potter; "Queen"+"Sussex" context → Prince Harry
- Initial embedding = blank canvas with general shape; attention paints the specific contextual meaning
- Without attention, every "tower" is the same tower, every "Harry" the same Harry

---

### SLIDE 03 — The mystery novel test

**Headline:** "Therefore, the murderer was ___" — that final word must encode the entire novel.

**Visual concept:** A book icon with the last line visible: "Therefore, the murderer was ___." Arrows flowing from various earlier points in the text (character introductions, clue scenes, red herrings) all converging on the word "was." The vector for "was" is shown glowing with accumulated information.

**Body:**
- Imagine a mystery novel where the last line is "Therefore, the murderer was..."
- The word "was" started as a generic past-tense verb
- By the final layer, its vector must encode the entire novel's worth of clues — every suspect, every alibi, every red herring
- The model needs to predict the murderer's name as the most probable next token
- Only attention can gather this information from across the full context window

**Key insight card:**
A single vector of 12,288 numbers, after 96 layers of attention, can encode an entire mystery novel's worth of reasoning. The simple verb "was" becomes a compressed summary of everything that matters.

**Speaker notes:**

- Mystery novel ending: "Therefore, the murderer was..." — model must predict the correct name
- "Was" entered as a simple past-tense verb; by final layer, attention compressed entire novel's clues into that one vector
- Ultimate attention test: attend to right info across thousands of tokens, ignore noise, compress to one prediction
- Current models do surprisingly well at this; longer contexts remain challenging

---

## Block 2: Queries, Keys, and the Attention Pattern (12 min)

### SLIDE 04 — The running example

**Headline:** "A fluffy blue creature roamed the verdant forest."

**Visual concept:** The sentence displayed with each word in its own box. The word "creature" is highlighted as the focus — it needs information from "fluffy" and "blue." The word "forest" is also highlighted — it needs information from "verdant."

**Body:**
- This is our running example for understanding how attention works step by step
- **Goal:** The word "creature" should absorb the meaning of "fluffy" and "blue"
- Similarly, "forest" should absorb "verdant"
- Each word needs to figure out: "Which other words are relevant to me?"

**Speaker notes:**

- Running example: "A fluffy blue creature roamed the verdant forest"
- Goal: "creature" absorbs fluffy+blue; "forest" absorbs verdant
- Challenge: how does "creature" know to attend to "fluffy"/"blue" but not "roamed"/"the" — without anyone programming "nouns look for adjectives"?
- Three learned matrices do this: query, key, value

---

### SLIDE 05 — Queries: "What am I looking for?"

**Headline:** Each word generates a query — a vector that represents what information it's seeking.

**Visual concept:** The word "creature" generating a query vector (shown as an arrow in a small space). The query is labeled: "Looking for: adjectives that describe me." Show the query matrix (W_Q) transforming the 12,288-dimensional embedding into a smaller 128-dimensional query vector.

**Body:**
- The **query matrix** (W_Q) transforms each word's embedding into a **query vector**
- The query encodes: "What kind of information am I looking for?"
- Nouns might generate queries that ask: "Are there adjectives describing me nearby?"
- Query vectors live in a smaller space — **128 dimensions** (down from 12,288)
- Each attention head has its own query matrix — learning to ask different questions

**Speaker notes:**

- Query matrix (W_Q) transforms each word's 12,288-D embedding into a 128-D query vector = "what information do I need?"
- Nouns learn to generate queries that match with adjective-type keys — not programmed, learned during training
- 128-D (not 12,288) is deliberate: cheaper computation and forces focus on relationship type
- Each attention head has its own W_Q — different heads ask different questions (adjectives, verbs, coreference, etc.)

---

### SLIDE 06 — Keys: "Here's what I have to offer"

**Headline:** Each word also generates a key — a vector advertising what information it can provide.

**Visual concept:** The words "fluffy" and "blue" each generating key vectors (shown as arrows in the same small space as the queries). Their keys are labeled: "Offering: I'm a descriptor / adjective." The key matrix (W_K) is shown transforming embeddings into the same 128-dimensional space as queries.

**Body:**
- The **key matrix** (W_K) transforms each word's embedding into a **key vector**
- The key encodes: "Here's what kind of information I have to offer"
- Adjectives like "fluffy" and "blue" generate keys that say: "I'm a descriptor!"
- Keys live in the **same 128-dimensional space** as queries — this is critical for matching
- Matching happens via **dot product**: query · key = relevance score

**Speaker notes:**

- Key matrix (W_K) transforms each word into a 128-D key vector = "here's what information I have to offer"
- Adjectives generate keys advertising "I'm a descriptor"; verbs advertise action info; "the" has minimal info to offer
- Keys live in same 128-D space as queries — compared via dot product: similar direction = relevant, different = not relevant
- "Creature" query (seeking descriptors) dot "fluffy" key (offering descriptor) = high score; dot "roamed" key = low score
- Matching is automatic — learned during training, not programmed

---

### SLIDE 07 — The attention pattern

**Headline:** Dot products between all queries and keys create a grid — the attention pattern.

**Visual concept:** A grid (matrix) where rows are queries (one per word) and columns are keys (one per word). Each cell is color-coded by the dot product score — bright = high relevance, dark = low. The cell where "creature" (query) meets "fluffy" (key) is bright. The cell where "creature" meets "roamed" is dark.

Below the grid: "Apply softmax along each row → scores sum to 1.0 → attention weights"

A triangular mask overlay on the upper-right portion, labeled: "Masking: later words can't attend to future words (set to -infinity before softmax)"

**Body:**
- Compute dot products between every query-key pair → a grid of relevance scores
- Apply **softmax** along each row to normalize scores into weights that sum to 1.0
- High weight = "I should pay a lot of attention to this word"
- Low weight = "I should mostly ignore this word"
- **Masking:** Words can only attend to words that came before them — future words are masked out (set to negative infinity before softmax, which makes them zero after)

**Speaker notes:**

- Dot product every query with every key → grid of relevance scores (rows = queries, columns = keys)
- Softmax each row → scores sum to 1.0 = attention weights; e.g., "creature" gives 40% to "fluffy," 35% to "blue," small amounts elsewhere
- Masking: words can't attend to future positions — set future scores to -infinity before softmax → become zero after
- Result is a triangular attention pattern — each word only attends to itself and earlier words

---

## Block 3: Values — The Actual Information Exchange (8 min)

### SLIDE 08 — Values: what information actually gets passed

**Headline:** A third matrix produces the value — the actual content that gets communicated.

**Visual concept:** The word "fluffy" generating three vectors:
- Query (what it's looking for)
- Key (what it offers — for matching)
- Value (the actual information to transmit — "here's the meaning of fluffiness to add to whoever is paying attention to me")

Show: "creature" paying 40% attention to "fluffy" → receives 40% of fluffy's value vector → that value gets ADDED to creature's embedding.

**Body:**
- Queries and keys determine **who** pays attention to **whom**
- The **value vector** determines **what information** actually gets transferred
- Value matrix (W_V) transforms each word into the information it can contribute
- The final update to each word = weighted sum of all value vectors, weighted by attention scores
- This update gets **added** to the original embedding — enriching it with context

**Speaker notes:**

- Queries/keys = matchmaking (who attends to whom); values = the actual content transferred
- Value matrix (W_V) encodes: "if someone attends to me, here's the info I give them" — "fluffy" offers soft/fuzzy/textured meaning
- Transfer: multiply each word's value vector by its attention weight, sum them all → context update for the attending word
- Update gets ADDED to original embedding — "creature" now encodes fluffiness and blueness; enriched by context

---

### SLIDE 09 — The value map factorization

**Headline:** The value transformation is factored into two smaller matrices — value-down and value-up.

**Visual concept:** A flow diagram:
- Embedding (12,288 dims) → **Value-down matrix** → small vector (128 dims) → **Value-up matrix** → update vector (12,288 dims) → ADD to original embedding

Label: "Low-rank factorization — ~1.5M parameters per matrix per head (instead of ~150M for a full-size matrix)"

**Body:**
- Instead of one huge value matrix, the transformation is split into two smaller ones
- **Value-down:** Compress from 12,288 dimensions to 128 dimensions
- **Value-up:** Expand back from 128 to 12,288 dimensions
- This is a **low-rank factorization** — it dramatically reduces parameter count
- Each matrix is ~1.5 million parameters per head
- The compressed intermediate step forces the model to extract only the most essential information

**Speaker notes:**

- Full 12,288→12,288 value matrix would be ~150M params per head — too expensive with 96 heads
- Solution: low-rank factorization — value-down (12,288→128, ~1.5M params) then value-up (128→12,288, ~1.5M params) = 50x savings
- 128-D bottleneck forces compression — only most essential information passes through
- Each head has its own value-down/value-up matrices, extracting and contributing different info types

---

## Block 4: Multi-Headed Attention and the Full Picture (10 min)

### SLIDE 10 — Multi-headed attention: 96 perspectives at once

**Headline:** GPT-3 runs 96 attention heads in parallel — each looking for different relationships.

**Visual concept:** A grid of 96 small attention-pattern squares, each with a different pattern. A few are enlarged with labels:
- Head 12: "Subject → Verb" (grammar tracking)
- Head 37: "Pronoun → Antecedent" (coreference)
- Head 61: "Adjective → Noun" (property attribution)
- Head 84: "Entity → Entity" (semantic relationships)

Below: all 96 value-up outputs summing together into one combined update per word.

**Body:**
- One attention head learns one type of relationship — but language has many types
- GPT-3: **96 heads per layer**, each with its own Q, K, V matrices
- Different heads learn different patterns:
  - Grammar (subject-verb agreement)
  - Pronoun resolution ("it" refers to which noun?)
  - Sentiment tracking
  - Semantic similarity
  - Positional relationships
- All 96 heads produce proposed updates — all summed together into one combined context update

**Speaker notes:**

- Single head = one type of relationship; language has many simultaneous relationship types (grammar, coreference, sentiment, etc.)
- GPT-3: 96 heads per layer, each with separate Q/K/V matrices learning different patterns independently
- Examples: subject-verb agreement, pronoun resolution ("it" → "animal" not "street"), entity disambiguation
- Nobody tells heads what to look for — self-organize during training to reduce prediction error
- All 96 heads produce proposed updates → summed into one combined context update per word (96 perspectives at once)

---

### SLIDE 11 — The output matrix

**Headline:** All the value-up matrices from all heads get combined into one output matrix.

**Visual concept:** 96 narrow vertical matrices (value-up, one per head) side by side, stapled together into one wide output matrix. An arrow showing this combined matrix producing the final attention output for one layer. Label: "Each head contributes its perspective. The output matrix combines them all."

**Body:**
- Each head's value-up matrix produces a proposed change to the embedding
- All 96 value-up matrices can be thought of as **one large output matrix**, stapled side by side
- The combined result: a single update vector per word that blends all 96 heads' contributions
- This update gets **added** to the original embedding
- Then the updated embedding flows into the MLP block for knowledge injection

**Speaker notes:**

- All 96 value-up matrices stapled into one large output matrix — one big matrix multiply instead of 96 separate ones (GPU efficient)
- Result: single update vector per word blending all 96 heads' contributions → added to original embedding
- Updated embedding flows into MLP block (attention = relationships, MLP = knowledge) — together = one Transformer layer
- Repeats 95 more times in GPT-3

---

### SLIDE 12 — Repeated layers: 96 rounds of refinement

**Headline:** GPT-3 repeats the attention + MLP cycle 96 times — each layer refining the representation further.

**Visual concept:** A vertical stack of 96 layer blocks, each containing an attention block and an MLP block. Early layers are labeled "surface patterns" (grammar, syntax). Middle layers are labeled "semantic relationships" (meaning, entities). Late layers are labeled "abstract reasoning" (inference, prediction). A parameter count callout: "Attention: ~58 billion (1/3). MLP: ~116 billion (2/3). Total: ~175 billion."

**Body:**
- 96 layers, each with attention + MLP
- **Early layers** tend to capture surface patterns — grammar, syntax, word order
- **Middle layers** build semantic understanding — entities, relationships, themes
- **Late layers** perform higher-level reasoning — inference, prediction, abstraction
- **Attention parameters:** ~58 billion (one-third of GPT-3)
- **MLP parameters:** ~116 billion (two-thirds of GPT-3)
- Each word's vector gets progressively enriched through all 96 layers

**Speaker notes:**

- 96 layers: early = grammar/syntax, middle = semantic understanding, late = abstract reasoning/inference
- 96 heads x 96 layers = 9,216 total attention operations — each word analyzed from 9,000+ perspectives
- Attention: ~58B params (1/3); MLP: ~116B params (2/3)
- By final layer, each word's vector has been updated from 9,216 perspectives + 96 MLP knowledge injections
- That's how "was" in "the murderer was" encodes an entire novel's worth of clues

---

### SLIDE 13 — Why attention succeeded

**Headline:** The secret ingredient wasn't just cleverness — it was parallelism.

**Visual concept:** Two contrasted approaches:
1. **Before Transformers (RNNs):** A sequential chain — word 1 → word 2 → word 3 → ... → word 2000. Label: "Must process sequentially. Slow. Information degrades over distance."
2. **Transformers (Attention):** A grid of all-to-all connections — every word connected to every other word simultaneously. Label: "All comparisons happen in parallel. GPU-friendly. Information flows freely."

**Body:**
- Before Transformers, language models (RNNs) processed text sequentially — one word at a time
- Information had to pass through a chain, degrading over long distances (the "vanishing gradient" problem)
- Attention computes all word-to-word relationships **simultaneously** — massively parallelizable on GPUs
- **Scale + parallelism = breakthrough**
- The mechanism is good, but the ability to make it enormous is what changed everything

**Key insight card:**
Attention didn't just change how AI understands language — it changed how efficiently AI can be trained. Parallel computation on GPUs meant you could scale from millions to billions to hundreds of billions of parameters. Scale is the enabler.

**Speaker notes:**

- Attention's success = parallelizability, not just cleverness
- Before: RNNs processed sequentially (word by word) — slow, info degrades over distance (vanishing gradient), can't parallelize
- Attention: every word attends to every other directly — all dot products computed simultaneously on GPU
- Scale + parallelism = breakthrough; Transformers turned language AI from serial bottleneck to parallel computation
- RNNs can't scale; Transformers can → billions of parameters, billions of examples → that unlocked everything

---

## Wrap-up (2 min)

### SLIDE 14 — Chapter 7 recap

**Headline:** Attention — the mechanism that lets words understand each other.

**Visual concept:** A summary diagram showing the full attention pipeline:
1. Embedding → Query (W_Q): "What am I looking for?"
2. Embedding → Key (W_K): "What do I offer?"
3. Query · Key → Attention pattern (with masking)
4. Softmax → Attention weights
5. Embedding → Value-down → Value-up: "What information to transfer"
6. Weighted sum of values → Add to embedding
7. Repeat across 96 heads, 96 layers

**Body:**
What we covered:
- **The problem:** Same word, different meanings — context is everything
- **Queries** ask "What am I looking for?" — learned per head
- **Keys** advertise "Here's what I offer" — matched to queries via dot product
- **Attention pattern:** Grid of relevance scores, softmax-normalized, with masking to prevent seeing the future
- **Values** carry the actual information to transfer — factored into value-down and value-up matrices
- **Multi-headed:** 96 heads per layer, each learning different relationship types
- **96 layers:** 9,216 total attention operations — over 9,000 perspectives per word
- **Why it works:** Massively parallelizable on GPUs — scale + parallelism = breakthrough

**Speaker notes:**

- Problem: identical embeddings regardless of context → attention fixes by letting words update each other
- Pipeline: query ("looking for?") + key ("I offer?") → dot product → softmax → attention weights → value transfer → add to embedding
- 96 heads in parallel x 96 layers = 9,216 attention operations per word
- Attention turns flat embeddings into rich contextual representations ("king" → "Scottish king in Macbeth")
- Attention = relationships; MLP = knowledge; together = the complete Transformer architecture

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| Block 1: The problem attention solves | 3 (slides 01-03) | 8 min |
| Block 2: Queries, keys, and attention pattern | 4 (slides 04-07) | 12 min |
| Block 3: Values | 2 (slides 08-09) | 8 min |
| Block 4: Multi-headed attention & full picture | 4 (slides 10-13) | 10 min |
| Wrap-up | 1 (slide 14) | 2 min |
| **Total** | **14 slides** | **~40 min** |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 6: ["Attention in transformers, step-by-step"](https://www.youtube.com/watch?v=eMlx5fFNoYc) (2024)
