# Chapter 6 — Transformers: The Tech Behind LLMs

## Presentation Guide

**Duration:** 35-40 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build genuine intuition for the Transformer architecture — the engine behind ChatGPT and Claude — covering tokenization, embeddings, the data flow pipeline, softmax, and temperature so that the attention deep-dive (Chapter 7) has a solid foundation.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 5: "But what is a GPT? Visual intro to Transformers"](https://www.youtube.com/watch?v=wjZofJX0v4M) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **Block 1: What is a GPT?** | 8 min |
| 0:08 | **Block 2: Tokenization and embeddings** | 10 min |
| 0:18 | **Block 3: Semantic meaning in embedding space** | 8 min |
| 0:26 | **Block 4: The pipeline — from embedding to prediction** | 12 min |
| 0:38 | **Wrap-up** | 2 min |

**Total: ~40 minutes**

---

## Block 1: What Is a GPT? (8 min)

### SLIDE 01 — GPT: three words, one architecture

**Headline:** GPT stands for Generative Pre-trained Transformer — and each word matters.

**Body:**
- **Generative** — it produces new text, not just classifies or labels
- **Pre-trained** — it learned from massive datasets before you ever typed a prompt
- **Transformer** — the specific neural network architecture that made the breakthrough (2017 Google paper: "Attention Is All You Need")

**Visual concept:** The letters G-P-T, each expanding into its full word with a one-line definition beneath. An arrow from "Transformer" pointing to a simplified block diagram of the architecture.

**Key insight card:**
Every major AI you've heard of — ChatGPT, Claude, Gemini, Llama — is a Transformer under the hood. Different companies, different training data, same fundamental blueprint.

**Speaker notes:**

- Generative = creates new text token by token; Pre-trained = billions of gradient steps already done; Transformer = the architecture
- 2017 Google paper "Attention Is All You Need" — arguably most consequential ML paper ever; enabled language AI to scale
- Every major model (ChatGPT, Claude, Gemini, Llama) uses this same architecture — different data/tuning, same blueprint

---

### SLIDE 02 — The high-level data flow

**Headline:** Text in, prediction out — here's the assembly line.

**Visual concept:** A vertical pipeline flowing top to bottom:
1. Raw text → **Tokenizer** → Token IDs
2. Token IDs → **Embedding matrix** → Vectors (one per token)
3. Vectors → **Attention block** (blue) → **MLP block** (green) → repeat ~96 times
4. Final vector → **Unembedding matrix** → Raw scores (logits)
5. Logits → **Softmax** → Probability distribution
6. Sample → **Next token prediction**

**Body:**
- Text gets chopped into tokens, converted to numbers
- Each token becomes a high-dimensional vector (list of numbers encoding meaning)
- Those vectors flow through dozens of alternating attention and MLP blocks
- The final vector gets converted back to a probability for every possible next word
- The model picks one, appends it, and repeats the whole process

**Speaker notes:**

- This slide is the roadmap — everything that follows unpacks one piece of this pipeline
- Text → numbers → 96 layers of processing → probability distribution over ~50,000 next words
- GPT-3 repeats attention+MLP pair 96 times; each pass refines understanding
- Model samples one word, appends it, restarts pipeline for the next word

---

### SLIDE 03 — Deep learning recap: the building blocks

**Headline:** Everything in deep learning boils down to four ideas.

**Body:**
1. **Input as numbers** — images become pixel arrays, text becomes token IDs, audio becomes waveform samples
2. **Layers transform data** — each layer takes numbers in and produces numbers out
3. **Weights interact via weighted sums** — matrix multiplication is the core operation
4. **Non-linear functions sprinkled in** — activation functions (like ReLU) prevent the whole network from collapsing into a single linear transformation

**Visual concept:** A simple flow: Input numbers → [Matrix multiply] → [Add bias] → [Non-linear squish] → Output numbers. Label: "This is one layer. Stack dozens of them."

**Speaker notes:**

- Four building blocks: (1) inputs as numbers, (2) layers = functions (numbers in/out), (3) weighted sums via matrix multiply (GPU-optimized), (4) non-linear activations (ReLU) prevent layer collapse
- Without non-linearity, stacking layers is pointless — multiple linear layers collapse into one
- All of deep learning is these four ideas composed at scale

---

### SLIDE 04 — GPT-3 by the numbers

**Headline:** 175 billion parameters, organized into ~28,000 matrices across 8 categories.

**Body:**
- **175 billion parameters** — adjustable numbers learned during training
- Organized into roughly **28,000 matrices**
- Those matrices fall into **8 categories** (embedding, unembedding, query, key, value, output, MLP up-projection, MLP down-projection)
- All learned via the same gradient descent process from Session 0 — just at staggering scale

**Visual concept:** A treemap or block diagram showing the 8 matrix categories with approximate parameter counts. The two largest blocks (MLP up and MLP down) visually dominate. A small callout: "Our digit recognizer had 13,002 parameters. GPT-3 has 13.5 million times more."

**Key insight card:**
The architecture is not mysterious. It's the same building blocks — matrix multiplication, bias, activation function — repeated at enormous scale. The complexity comes from the *size*, not the *kind* of math.

**Speaker notes:**

- Our digit network: 13,002 parameters; GPT-3: 175 billion — 13.5 million times bigger, same math
- 175B parameters organized into ~28,000 matrices falling into just 8 categories
- 8 matrix types: embedding, unembedding, query, key, value, output, MLP up-projection, MLP down-projection
- Architecture is highly repetitive — learn the pattern once and you understand the whole thing

---

## Block 2: Tokenization and Embeddings (10 min)

### SLIDE 05 — Tokenization: breaking text into pieces

**Headline:** AI doesn't read words — it reads tokens.

**Visual concept:** The sentence "To date, the cleverest thinker of all time was..." broken apart with visible boundaries: `To` | ` date` | `,` | ` the` | ` cle` | `ver` | `est` | ` thinker` | ` of` | ` all` | ` time` | ` was` | `...`
Below: each token mapped to a numeric ID (e.g., "To" → 2514, " date" → 3128, etc.)

**Body:**
- Text gets split into chunks called **tokens** — usually whole words or word fragments
- Common words ("the", "and") are single tokens
- Longer/rarer words get split: "cleverest" → "cle" + "ver" + "est"
- GPT-3's vocabulary: **~50,257 tokens**
- Each token gets a unique numeric ID
- The model never sees raw text — only these IDs

**Speaker notes:**

- Tokenization = chopping text into chunks; common words are single tokens, rare/long words get split ("cleverest" → "cle"+"ver"+"est")
- Vocabulary: ~50,257 tokens, each with a unique numeric ID — model never sees raw text after this step
- Context window = max tokens processed at once: GPT-3 = 2,048; modern models = 100,000+

---

### SLIDE 06 — The embedding matrix: giving tokens meaning

**Headline:** Each token ID gets looked up in a giant table — the embedding matrix (W_E).

**Visual concept:** A large table (matrix) with 50,257 rows and 12,288 columns. One row is highlighted — that row IS the embedding vector for a particular token. Show the token "king" being looked up, producing a column of 12,288 numbers. Label the matrix: "W_E — ~617 million parameters."

**Body:**
- The **embedding matrix** (W_E) has one column for every token in the vocabulary
- Each column is a vector of **12,288 numbers** (for GPT-3)
- That vector encodes the meaning of the token in high-dimensional space
- Size: 50,257 tokens × 12,288 dimensions = **~617 million parameters**
- These numbers aren't hand-coded — they're **learned during training**

**Speaker notes:**

- Embedding matrix (W_E) = lookup table: each token ID maps to a vector of 12,288 numbers defining its position in high-dimensional space
- Size: 50,257 tokens x 12,288 dims = ~617M parameters — less than 0.5% of GPT-3's total 175B
- Embeddings start random, self-organize during training via gradient descent — the structure that emerges is remarkable

---

### SLIDE 07 — Word embeddings and semantic meaning

**Headline:** Similar words land near each other. Directions encode concepts.

**Visual concept:** A 2D simplified map showing:
- Cluster of royalty words: "king", "queen", "monarch", "ruler" — close together
- Arrow from "man" to "woman" labeled "gender direction"
- Same arrow from "king" to "queen"
- Arrow from "Germany" to "Italy" labeled "nationality direction"
- Same arrow from "Hitler" to "Mussolini"
- Separate example: "cats" minus "cat" = "plurality direction"

**Body:**
- Words with similar meanings cluster together: "king" near "queen" near "monarch"
- **Directions** in embedding space represent concepts:
  - woman − man ≈ queen − king (gender direction)
  - Italy − Germany + Hitler ≈ Mussolini (nationality + historical role)
  - cats − cat = a "plurality direction" that aligns with all plural nouns
- The network learned all of this from text alone — no human labeled anything

**Key insight card:**
Nobody programmed "king = male, royal, singular." The model discovered these abstract concepts by reading billions of sentences and noticing which words appear in similar contexts. The structure of human language is encoded as geometry.

**Speaker notes:**

- woman - man = gender direction; add it to "king" → land near "queen"; add to "uncle" → "aunt"
- Italy - Germany = nationality direction; Hitler + that direction → lands near Mussolini
- cats - cat = plurality direction; dot product with "one"/"two"/"three"/"four" gives increasing values
- Nobody labeled any of this — emerged from training; words in similar contexts cluster; language structure becomes geometry in 12,288-D space

---

### SLIDE 08 — Dot products as similarity

**Headline:** How does the model measure whether two things are related? The dot product.

**Visual concept:** Three pairs of arrows:
1. Two arrows pointing the same direction → dot product = large positive number → "aligned / similar"
2. Two arrows at 90 degrees → dot product = 0 → "unrelated"
3. Two arrows pointing opposite directions → dot product = large negative number → "opposite"

**Body:**
- The **dot product** of two vectors measures how aligned they are
- **Positive** = pointing in similar directions (related concepts)
- **Zero** = perpendicular (unrelated)
- **Negative** = pointing in opposite directions (opposing concepts)
- This one operation — the dot product — is the engine behind attention, similarity search, and almost every comparison the model makes

**Speaker notes:**

- Dot product = single most important operation in Transformers; measures how aligned two vectors are
- Same direction → large positive (similar); perpendicular → zero (unrelated); opposite → large negative (opposites)
- Used everywhere: word similarity, query-key matching in attention, next-word scoring in unembedding
- Fast and parallelizable — GPUs compute millions of dot products simultaneously

---

## Block 3: Semantic Meaning in Embedding Space (8 min)

### SLIDE 09 — Context size and the journey through the network

**Headline:** Each vector starts generic and ends up encoding rich, specific meaning.

**Visual concept:** A timeline showing the word "king" at different stages:
- **Entering the network:** "king" = generic royalty concept
- **After early layers:** "king" + "Scottish" context
- **After middle layers:** "king" + "murdered predecessor" + "Shakespeare"
- **After final layers:** "A specific Scottish king who murdered his predecessor, in a Shakespeare play from ~1606, and the text is about to discuss psychological consequences"

**Body:**
- GPT-3's context window: **2,048 tokens** — the maximum text the model can "see" at once
- Embedding vectors are meant to **soak up context** as they flow through the network
- Layer by layer, attention pulls in surrounding information
- By the final layer, a single vector encodes far more than just the dictionary meaning of its word

**Speaker notes:**

- Context window: GPT-3 = 2,048 tokens; modern models much larger but same principle
- Vectors don't stay static — start as generic embeddings, get progressively enriched through 96 layers of attention
- Example: "king" starts as generic royalty → after attention in Macbeth context becomes "Scottish king who murdered predecessor, Shakespeare, ~1606, building toward psychological consequences"
- All encoded in a single 12,288-number vector — progressive context enrichment is what makes Transformers powerful

---

### SLIDE 10 — The unembedding matrix

**Headline:** At the end, the model converts the final vector back into a prediction.

**Visual concept:** A vector (12,288 numbers) entering a matrix labeled "W_U — Unembedding matrix (~617M parameters)" and producing a bar chart of ~50,257 scores — one per vocabulary token. The highest bars labeled with words like "Paris," "the," "a."

**Body:**
- The **unembedding matrix** (W_U) maps the final vector back to ~50,257 scores — one per token in the vocabulary
- It's the reverse of embedding: embedding turns words into vectors, unembedding turns vectors back into word scores
- Size: **~617 million parameters** (same as the embedding matrix)
- Combined embedding + unembedding: **~1.2 billion parameters**
- The raw output scores are called **logits** — they're not yet probabilities

**Speaker notes:**

- Unembedding matrix (W_U) = reverse of embedding: multiply final vector → ~50,257 scores, one per token; higher = more likely next
- ~617M parameters; combined with embedding = ~1.2B just for input/output conversion; remaining ~174B in the 96 attention+MLP layers
- Raw output scores = logits (arbitrary numbers, positive or negative) — not yet probabilities; need softmax to convert

---

### SLIDE 11 — Softmax: turning scores into probabilities

**Headline:** Softmax turns a list of arbitrary numbers into a valid probability distribution.

**Visual concept:** A list of raw logits on the left: [3.2, 1.1, 0.3, -0.5, 5.8, ...] with an arrow labeled "softmax" pointing to a bar chart on the right where values sum to 1.0. The bar for 5.8 dominates. Label: "Largest values dominate. Small values get squished toward zero."

**Body:**
- Raw logits can be any number — positive, negative, large, small
- **Softmax** converts them into probabilities that sum to 1.0
- How it works: raise *e* to the power of each logit, then divide by the total
- The largest values dominate — softmax amplifies the gaps between scores
- The result: a probability distribution across all ~50,257 possible next tokens

**Speaker notes:**

- Softmax: raise e to each logit, divide by total — converts arbitrary numbers into probabilities summing to 1.0
- Key behavior: amplifies differences — largest logit dominates the distribution; small logit gaps become big probability gaps
- After softmax: probability for every token ("Paris" 68%, "the" 4%, "a" 3%) — model samples from this distribution

---

### SLIDE 12 — Temperature: controlling randomness

**Headline:** Temperature controls how creative vs. predictable the model is.

**Visual concept:** Three probability distributions side by side:
1. **T = 0 (cold):** One bar dominates at ~95%, everything else near zero. Label: "Always picks the most likely word. Deterministic but repetitive."
2. **T = 1 (default):** A spread of bars, one still tallest but others visible. Label: "Balanced — confident but willing to surprise."
3. **T = high (hot):** Bars are nearly equal height. Label: "Everything is equally likely. Creative but chaotic."

Below: "Once upon a time" example — T=0 produces a Goldilocks derivative, high T produces "South Korean web artist" degenerating into nonsense.

**Body:**
- Before softmax, divide all logits by the **temperature** value
- **T = 0:** Always pick the most likely word → predictable, safe, but boring and repetitive
- **High T:** Flatten the distribution → more creative but risky, can degenerate into nonsense
- **Example:** "Once upon a time..."
  - T = 0 → always generates a Goldilocks derivative (the obvious, boring completion)
  - High T → might generate "a South Korean web artist..." then degenerate into incoherent text

**Analogy card:**
Temperature is like a confidence dial. Turn it down and the model only picks safe, obvious answers. Turn it up and it's willing to take risks — sometimes brilliantly creative, sometimes nonsensical.

**Speaker notes:**

- Mechanically: divide all logits by T before softmax; low T amplifies gaps (top choice dominates), high T flattens (all choices equalize)
- T=0: deterministic, always picks most likely token — safe but repetitive; "Once upon a time" always → Goldilocks variant
- High T: picks lower-probability tokens — creative but can spiral into nonsense ("South Korean web artist" → incoherent)
- Why ChatGPT sometimes gives same answer twice, sometimes different — temperature controls sampling randomness
- Production systems use moderate T — creative enough to be useful, not so much it's unreliable

---

## Block 4: The Pipeline — From Embedding to Prediction (12 min)

### SLIDE 13 — The Transformer pipeline: attention + MLP, repeated

**Headline:** The same two operations, repeated 96 times.

**Visual concept:** A vertical pipeline showing tokens entering at the bottom, flowing upward through clearly labeled alternating blocks:
- ATTENTION BLOCK 1 (blue) — "Words talk to each other"
- MLP BLOCK 1 (green) — "Facts get injected"
- ATTENTION BLOCK 2 (blue)
- MLP BLOCK 2 (green)
- ... (dotted lines indicating repetition)
- ATTENTION BLOCK 96 (blue)
- MLP BLOCK 96 (green)
- UNEMBEDDING → SOFTMAX → PREDICTION

**Body:**
- **Attention blocks:** Each word looks at every other word and updates its meaning based on context
- **MLP blocks:** Process each word independently — inject knowledge and refine understanding
- These two operations alternate, 96 times in GPT-3
- After 96 layers: unembedding → softmax → next-token prediction

**Speaker notes:**

- Same pair of operations (attention → MLP) repeated 96 times; each pass refines understanding
- Attention = words communicate with each other ("which words are relevant to my meaning?") — context incorporation
- MLP = factual knowledge injection per word independently ("Michael Jordan? → add basketball, Bulls, #23") — 2/3 of parameters live here
- After 96 rounds: unembedding → softmax → probability distribution over ~50K tokens → sample one → repeat pipeline

---

### SLIDE 14 — Parameter budget: where do 175 billion numbers live?

**Headline:** A breakdown of GPT-3's parameter budget.

**Visual concept:** A stacked bar chart or pie chart:
- **Embedding matrix (W_E):** ~617M (0.35%)
- **Unembedding matrix (W_U):** ~617M (0.35%)
- **Attention parameters:** ~58B (33%) — query, key, value, output matrices across 96 layers × 96 heads
- **MLP parameters:** ~116B (66%) — up-projection and down-projection matrices across 96 layers

**Body:**
| Component | Parameters | Share |
|-----------|-----------|-------|
| Embedding (W_E) | ~617 million | ~0.35% |
| Unembedding (W_U) | ~617 million | ~0.35% |
| Attention (Q, K, V, O) | ~58 billion | ~33% |
| MLP (up + down projection) | ~116 billion | ~66% |
| **Total** | **~175 billion** | **100%** |

**Speaker notes:**

- Embedding + unembedding: ~1.2B combined (<1%) — dictionary layers converting words ↔ vectors
- Attention (Q, K, V, O across 96 layers x 96 heads): ~58B (33%) — context understanding
- MLP (up/down projection across 96 layers): ~116B (66%) — factual knowledge storage
- Surprise: most of the model is MLP, not attention — attention gets headlines but MLP stores knowledge
- All 175B numbers learned via gradient descent — same algorithm from Session 0, mind-boggling scale

---

## Wrap-up (2 min)

### SLIDE 15 — Chapter 6 recap

**Headline:** From text to prediction — the Transformer assembly line.

**Visual concept:** A horizontal flow diagram summarizing the full pipeline:
Text → Tokenize → Embed (W_E) → [Attention + MLP] × 96 → Unembed (W_U) → Softmax → Predict next token → Repeat

**Body:**
What we covered:
- **GPT** = Generative Pre-trained Transformer
- **Tokenization** breaks text into ~50,257 possible pieces
- **Embeddings** turn tokens into 12,288-dimensional vectors where meaning is encoded as geometry
- **Dot products** measure similarity between vectors
- **Context** enriches each vector as it flows through 96 layers
- **Unembedding + softmax** converts the final vector back into a next-word prediction
- **Temperature** controls randomness: cold = safe, hot = creative
- **175 billion parameters** organized into 8 matrix types across ~28,000 matrices

**Speaker notes:**

- Full pipeline: text → tokenize → embed (12,288-D vectors) → 96 layers attention+MLP → unembed → softmax → sample next token
- Key numbers: 50,257 tokens, 12,288 dimensions, 96 layers, 175 billion parameters
- Temperature controls randomness in token selection
- Next chapter: deep-dive on attention — the mechanism that lets words understand context

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| Block 1: What is a GPT? | 4 (slides 01-04) | 8 min |
| Block 2: Tokenization & embeddings | 4 (slides 05-08) | 10 min |
| Block 3: Semantic meaning | 4 (slides 09-12) | 8 min |
| Block 4: The pipeline | 2 (slides 13-14) | 12 min |
| Wrap-up | 1 (slide 15) | 2 min |
| **Total** | **15 slides** | **~40 min** |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 5: ["But what is a GPT? Visual intro to Transformers"](https://www.youtube.com/watch?v=wjZofJX0v4M) (2024)
