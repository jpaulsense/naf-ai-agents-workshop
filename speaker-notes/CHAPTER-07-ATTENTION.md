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

- The fundamental problem: initial embeddings are context-free
  - The word "mole" gets the exact same 12,288 numbers every time — whether it means an animal, a chemistry unit (6.022 times 10 to the 23rd), or a skin growth
  - Three completely different meanings, identical starting vectors
- The network needs a way for surrounding words to reach over and update each other's meanings
  - "American shrew" should tell "mole" to emphasize its animal-related dimensions
  - "One... of carbon dioxide" should push "mole" toward its chemistry meaning
  - "biopsy" should activate the medical meaning
- This is the problem attention was invented to solve
  - It's the mechanism that lets words look at each other and update their embeddings based on context

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

- Grant uses great examples to drive this home
  - "Eiffel tower" vs. "miniature tower" — the word "tower" starts identical in both cases
    - But "Eiffel" should inject: Paris, wrought iron, 1,000 feet tall, built in 1889
    - "Miniature" should inject: small, decorative, maybe a toy
  - "Harry" is completely ambiguous on its own
    - Add "wizard" and surrounding Harry Potter context → it's the boy who lived
    - Add "Queen" and "Sussex" → it's Prince Harry, Duke of Sussex
- The initial embedding is like a blank canvas
  - It has the general shape of the word's meaning
  - But attention paints the specific, contextual meaning based on what surrounds it
- This is why attention is so critical — without it, every "tower" would be the same tower, every "Harry" the same Harry

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

- This is Grant's most powerful example of why attention matters
  - Imagine a mystery novel — hundreds of pages of suspects, clues, misdirection, alibis
  - The last line: "Therefore, the murderer was..."
  - The model needs to predict the correct name
- The word "was" entered the network as a simple past-tense verb
  - But by the time it reaches the final layer, attention has pulled in information from across the entire context
  - Every relevant clue, every character introduction, every red herring has been gathered and compressed into that single vector
- This is the ultimate test of attention: can the model attend to the right information across thousands of tokens, ignore the noise, and compress the answer into a single vector that predicts the correct next word?
- In practice, current models can do surprisingly well at this — though longer contexts remain challenging

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

- We'll walk through the attention mechanism using this sentence from Grant's video
  - "A fluffy blue creature roamed the verdant forest"
- The goal of attention in this sentence:
  - "Creature" should end up knowing it's fluffy and blue
  - "Forest" should end up knowing it's verdant (lush and green)
- The challenge: how does "creature" know to pay attention to "fluffy" and "blue" but not "roamed" or "the"?
  - And how does this work automatically, without anyone programming "nouns should look for adjectives"?
- Three learned matrices make this happen: **query**, **key**, and **value**
  - Let's take them one at a time

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

- First piece of the puzzle: **queries**
  - Every word in the sentence gets multiplied by a query matrix (W_Q) to produce a query vector
  - Think of the query as the word raising its hand and asking: "What kind of information do I need?"
- For nouns like "creature," the query might encode something like: "Are there any adjectives sitting near me that describe what kind of creature I am?"
  - The model doesn't literally think in English — the query is a learned vector pattern
  - But the *effect* is that nouns learn to generate queries that match with adjective-type information
- The query vector is smaller than the full embedding — 128 dimensions instead of 12,288
  - This compression is deliberate — it makes the computation cheaper
  - And it forces the query to focus on the *type* of relationship, not every possible detail
- Each attention head has its own W_Q matrix, so different heads ask different questions
  - One head's queries might focus on "find my adjectives"
  - Another head's queries might focus on "find the verb I'm the subject of"

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

- Second piece: **keys**
  - Every word also gets multiplied by a key matrix (W_K) to produce a key vector
  - The key is the word's advertisement: "Here's what I have to offer"
- Adjectives like "fluffy" and "blue" generate keys that encode something like: "I'm a descriptor — I have property information to share"
  - "Roamed" generates a key that says something different — maybe "I'm a verb with action information"
  - "The" generates a key with minimal information to offer
- The critical design choice: keys live in the **same** 128-dimensional space as queries
  - This means you can directly compare them via dot product
  - When a query and key point in similar directions → high dot product → "these are relevant to each other"
  - When they point in different directions → low dot product → "not relevant"
- So "creature"'s query (looking for descriptors) will have a high dot product with "fluffy"'s key (offering descriptor information) and "blue"'s key
  - But low dot product with "roamed"'s key or "the"'s key
  - The matching happens automatically — learned during training

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

- Now we put queries and keys together
  - For every pair of words, compute the dot product of one word's query with another word's key
  - This creates a grid — rows are the words asking questions (queries), columns are the words offering answers (keys)
  - Each cell contains a score: how relevant is this key to this query?
- Apply softmax along each row to normalize
  - The scores in each row now sum to 1.0 — they become attention weights
  - "Creature" might give 40% attention to "fluffy," 35% to "blue," 10% to "A," and small amounts to everything else
- **Masking** is a critical detail for text generation
  - During training, the model processes entire sequences at once for efficiency
  - But a word at position 50 can't be allowed to look at position 51 — that's seeing the future
  - Solution: set all "future" scores to negative infinity before softmax
  - After softmax, negative infinity becomes zero — the word can't attend to anything after it
  - The attention pattern ends up looking like a triangle — each word only attends to itself and earlier words

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

- Queries and keys are about matchmaking — figuring out which words are relevant to which
  - But they don't carry the actual information that gets transferred
  - That's the job of the third matrix: the **value matrix** (W_V)
- Each word gets multiplied by the value matrix to produce a value vector
  - This vector encodes: "If someone is paying attention to me, here's the actual information I'll give them"
  - "Fluffy"'s value vector encodes the semantic content of fluffiness
  - Not "I'm an adjective" (that was the key's job) but the actual meaning — soft, fuzzy, textured
- The information transfer works like this:
  - "Creature" has attention weights: 40% on "fluffy," 35% on "blue," 10% on "A," etc.
  - Multiply each word's value vector by the attention weight
  - Sum them all up → this is the context update for "creature"
  - Add that update to "creature"'s original embedding
- After this step, "creature" is no longer a generic creature
  - Its embedding has been nudged in the direction of fluffiness and blueness
  - It's been enriched by context

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

- A practical detail about how values are computed
  - A full value matrix mapping 12,288 dimensions to 12,288 dimensions would have ~150 million parameters — per head
  - With 96 heads, that's 14.4 billion parameters in just the value matrices of one layer
  - Way too expensive
- Solution: factor the value transformation into two smaller matrices
  - **Value-down**: compress from 12,288 to 128 dimensions (~1.5M parameters)
  - **Value-up**: expand from 128 back to 12,288 dimensions (~1.5M parameters)
  - Total: ~3 million parameters instead of ~150 million — a 50x savings
- This is called a low-rank factorization
  - The bottleneck at 128 dimensions forces the model to compress — to extract only the most important bits
  - It's like squeezing information through a narrow pipe: only the most important content gets through
- Each attention head has its own value-down and value-up matrices
  - So different heads extract and contribute different types of information

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

- What we've described so far is a single attention head — one set of Q, K, V matrices learning one type of relationship
  - But language has many types of relationships happening simultaneously
  - Grammar, coreference, sentiment, topic tracking, temporal ordering, logical connections...
- So the model runs 96 heads in parallel, each with completely separate Q, K, V matrices
  - Each head independently learns to look for a different type of pattern
  - One head might learn to connect subjects with verbs
  - Another might learn pronoun resolution — figuring out that "it" in "The animal didn't cross the street because it was too tired" refers to "animal," not "street"
  - Another might track whether we're talking about Harry Potter or Prince Harry
- Nobody tells the heads what to look for — they self-organize during training
  - Each head finds relationship patterns that help reduce prediction error
- The output: all 96 heads produce their own proposed update to each word's embedding
  - These updates all get summed together into one combined context update per word
  - So each word is enriched from 96 different perspectives simultaneously

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

- Implementation detail: all 96 value-up matrices are effectively stapled together into one large output matrix
  - Rather than 96 separate matrix multiplications followed by a sum, it's one big matrix multiplication
  - Mathematically equivalent, but more efficient on GPUs
- The result is a single update vector per word
  - This vector blends all 96 heads' contributions — grammar, coreference, sentiment, everything
  - It gets added to the original embedding
- Then the updated embedding flows into the MLP block
  - Remember: attention handles relationships between words, MLP handles factual knowledge
  - These two operations together form one Transformer layer
- Then the whole thing repeats — another 95 times in GPT-3

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

- The full pipeline: 96 layers, each with attention then MLP
  - Each layer builds on the output of the previous one
  - Early layers tend to handle simpler patterns — grammar, basic syntax
  - Middle layers build up semantic understanding — who's who, what's related to what
  - Later layers handle more abstract reasoning — inference, logical connections, prediction
- The scale is staggering
  - 96 heads per layer times 96 layers = **9,216 total attention operations**
  - Each word gets analyzed from over 9,000 different perspectives as it flows through the network
  - About 58 billion parameters are devoted to attention — one-third of the model
  - About 116 billion parameters are in the MLP layers — two-thirds of the model
- By the time a word reaches the final layer, its embedding has been updated by every relevant word in the context, from 9,216 different perspectives, and enriched with factual knowledge from 96 MLP blocks
  - That's how "was" in "the murderer was" can encode an entire mystery novel's worth of clues

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

- Grant makes a crucial point: attention's success isn't just about the mechanism being clever
  - It's about the mechanism being **parallelizable**
- Before Transformers, the dominant language architecture was the RNN (recurrent neural network)
  - RNNs process words one at a time, sequentially — word 1 feeds into word 2, which feeds into word 3...
  - Information had to travel through the entire chain to connect distant words
  - Over long sequences, information degraded — the "vanishing gradient" problem
  - And because it's sequential, you can't parallelize it well — each step depends on the previous one
- Attention flips this completely
  - Every word can attend to every other word directly — no chain, no degradation
  - All the dot products (query times key) can be computed simultaneously on a GPU
  - This is why Transformers can scale to billions of parameters and train on billions of examples
- The insight: **scale + parallelism = breakthrough**
  - The attention mechanism is elegant, but what made it world-changing is that GPUs can run it at enormous scale
  - You couldn't do this with RNNs — they're inherently sequential
  - Transformers turned language AI from a serial bottleneck into a parallel computation — and that unlocked everything

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

- Quick recap of the full attention mechanism
  - The problem: words start with identical embeddings regardless of context — attention fixes that
  - Each word generates a query (what it's looking for) and a key (what it offers)
  - Dot products between queries and keys produce relevance scores
  - Softmax normalizes those scores into attention weights
  - Value vectors carry the actual information — weighted by attention scores, then added to the original embedding
  - 96 heads run in parallel, each learning different relationship types
  - 96 layers repeat the whole process — 9,216 total attention operations
- The big picture: attention is what turns flat, context-free embeddings into rich, contextual representations
  - It's how "king" becomes "a specific Scottish king from Macbeth"
  - It's how "was" in a mystery novel encodes the entire plot's worth of clues
  - It's the core innovation that made modern AI possible
- Combined with the MLP layers (which store factual knowledge), attention forms the complete Transformer architecture
  - Attention handles relationships. MLPs handle knowledge. Together, they produce intelligence.

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
