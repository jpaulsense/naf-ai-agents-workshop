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

- GPT is an acronym — break it down so the letters actually mean something
  - **Generative**: it creates new text, token by token
  - **Pre-trained**: billions of gradient descent steps already happened before you ever opened the app
  - **Transformer**: the architecture — the wiring diagram — that made large-scale language AI possible
- The Transformer architecture came from a 2017 Google paper titled "Attention Is All You Need"
  - Arguably the most consequential machine learning paper ever published
  - Before Transformers, language AI existed but couldn't scale — Transformers changed that
- Every major AI model — ChatGPT, Claude, Gemini, Llama — is built on this same architecture
  - Different training data, different fine-tuning, different companies
  - Same fundamental blueprint under the hood

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
  - Don't worry about memorizing it now; we'll walk through each step
- The key takeaway: text goes in as words, gets converted to numbers, flows through many layers of processing, and comes out as a prediction for the next word
- GPT-3 repeats the attention+MLP pair 96 times
  - Each pass refines the model's understanding of what the text means and what should come next
- After all 96 layers, the final vector gets converted back into a probability distribution across ~50,000 possible next words
  - The model samples one word, appends it, and starts the whole pipeline again for the next word

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

- Quick prerequisite recap — four ideas that underpin everything
  - Every input gets converted to an array of numbers — pixels, audio samples, token IDs, doesn't matter
  - Layers are just functions: numbers in, numbers out
  - The core math operation is the weighted sum — multiply each input by a weight, add them up
    - This is matrix multiplication — GPUs are built to do this fast
  - Non-linear activation functions (like ReLU) are critical
    - Without them, stacking layers would be pointless — multiple linear layers collapse into a single linear layer
    - The non-linearity is what lets deep networks learn complex patterns
- If any of this feels fuzzy from Session 0, that's fine — the concepts will reinforce as we go

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

- Put the scale in perspective
  - Our handwritten digit network: 13,002 parameters
  - GPT-3: 175 billion — that's 13.5 million times bigger
- But the math is the same — weighted sums, biases, activation functions, gradient descent
  - Nothing fundamentally new, just vastly more of it
- The 175 billion parameters live in about 28,000 matrices
  - Those matrices fall into 8 categories — we'll meet each one as we walk through the architecture
  - Embedding, unembedding, query, key, value, output, MLP up-projection, MLP down-projection
- The fact that it's organized into just 8 types of matrix is actually reassuring
  - It means the architecture is highly repetitive — learn the pattern once, and you understand the whole thing

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

- Neural networks only understand numbers, so step one is converting text to numbers
  - The process is called **tokenization** — chopping text into chunks called tokens
- Tokens are usually whole words, but not always
  - Common words like "the" or "of" are single tokens
  - Less common or longer words get split — "cleverest" becomes three tokens: "cle" + "ver" + "est"
  - This keeps the vocabulary manageable (~50,257 tokens) while handling any text
- Each token has a unique ID number — like a barcode
  - From this point forward, the model works purely with these ID numbers, never with raw text
- There's a limit on how many tokens the model can process at once — the **context window**
  - GPT-3: 2,048 tokens
  - Modern models (GPT-4, Claude): 100,000+ tokens
  - But the concept is the same — a fixed window of text the model can "see"

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

- Once text is tokenized, each token ID gets looked up in a giant table called the **embedding matrix**
  - Think of it as a dictionary where every word maps to a list of 12,288 numbers
  - Those 12,288 numbers define where that word lives in a high-dimensional space
- For GPT-3, the embedding matrix has about 617 million parameters
  - 50,257 tokens times 12,288 dimensions per token
  - That's over half a billion numbers — just for converting words to vectors
  - And that's less than half a percent of the model's total 175 billion parameters
- Nobody programs these embeddings by hand
  - They start as random numbers and self-organize during training
  - Billions of gradient descent steps shape them into a meaningful structure
  - The structure that emerges is remarkable — which we'll see on the next slide

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

- This is one of the most remarkable results in machine learning
  - Take the vector for "woman," subtract the vector for "man" — you get a direction that represents gender
  - Add that direction to "king" and you land near "queen"
  - Add it to "uncle" and you land near "aunt"
  - The model discovered that gender is a consistent direction in its embedding space
- It works for other concepts too
  - "Italy" minus "Germany" gives a direction representing Italian-ness vs. German-ness
  - Take "Hitler" (associated with Germany), add the Italy direction — land near "Mussolini"
  - The model has encoded nationality and historical role as separate directions
- Even plurality has a direction
  - "cats" minus "cat" produces a vector that, when you dot-product it with "one," "two," "three," "four," gives *increasing* values
  - The model learned that "four" is more plural than "one"
- Nobody hand-labeled any of this — it all emerged from training on text
  - Words that appear in similar contexts end up near each other
  - The structure of human language gets encoded as geometry in 12,288-dimensional space

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

- The dot product is the single most important operation in Transformers
  - It measures how aligned two vectors are — how much they point in the same direction
- Intuition:
  - Two vectors pointing the same way → large positive dot product → "these are similar"
  - Perpendicular vectors → dot product near zero → "these are unrelated"
  - Opposite directions → large negative dot product → "these are opposites"
- This is how the model compares anything to anything
  - "Is this word related to that word?" → dot product of their embeddings
  - "Does this query match that key?" → dot product (we'll see this in attention)
  - "How likely is this next word?" → dot product in the unembedding step
- It's fast, parallelizable, and mathematically elegant
  - GPUs can compute millions of dot products simultaneously

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

- GPT-3 can process up to 2,048 tokens at a time — that's its context window
  - Modern models have much larger windows, but the principle is the same
- The key idea: vectors don't stay static as they flow through the 96 layers
  - They start as generic dictionary-style embeddings
  - Each attention layer lets surrounding words update and enrich the vector
  - By the end, a vector that started as "king" might encode an incredibly specific meaning
- Grant's example: the word "king" entering a passage about Macbeth
  - Starts as a generic royalty concept
  - After attention pulls in context: becomes a Scottish king, who murdered his predecessor, written by Shakespeare, circa 1606, and the passage is building toward the psychological consequences
  - All encoded in a single vector of 12,288 numbers
- This is what makes Transformers powerful — the ability to progressively enrich meaning through context

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

- After 96 layers of attention and MLP processing, we have a final vector — 12,288 numbers packed with contextual meaning
  - Now we need to convert that back into a prediction: "What word comes next?"
- The unembedding matrix does this
  - It's essentially the reverse of the embedding step
  - Multiply the final vector by this matrix and you get ~50,257 scores, one per vocabulary token
  - Higher score = the model thinks that token is more likely to come next
- The unembedding matrix is another ~617 million parameters
  - Combined with the embedding matrix, that's about 1.2 billion parameters just for the input/output conversion
  - The remaining ~174 billion parameters are in the 96 layers of attention and MLP blocks
- The raw scores coming out of the unembedding matrix are called **logits**
  - They're not yet probabilities — they can be any number, positive or negative
  - To turn them into probabilities, we need one more step: softmax

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

- Logits are raw scores — they can be any number
  - They're not probabilities yet — they don't sum to 1, and they can be negative
- Softmax fixes that — it converts any list of numbers into a valid probability distribution
  - Every output is between 0 and 1
  - All outputs sum to exactly 1.0
- The key behavior: softmax amplifies differences
  - If one logit is much larger than the rest, it dominates the probability distribution
  - Small differences in logits can translate to big differences in probability
  - This is by design — the model is usually fairly confident about what comes next
- After softmax, you have a probability for every token in the vocabulary
  - "Paris" might be 68%, "the" might be 4%, "a" might be 3%, and so on
  - The model samples from this distribution to pick the next token

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

- Temperature is a single number that controls how random the model's choices are
  - Mechanically: divide all logits by T before applying softmax
  - Low T → differences between logits get amplified → the top choice dominates
  - High T → differences get flattened → all choices become more equally likely
- At T = 0, the model is completely deterministic
  - It always picks the single most likely next token
  - Safe and consistent, but repetitive and boring
  - Grant's example: "Once upon a time" at T=0 always produces some variant of Goldilocks — the most statistically common fairy tale opening
- At high temperature, the model takes risks
  - It's more willing to pick lower-probability tokens
  - Can produce creative, surprising text
  - But can also spiral into nonsense — Grant's example starts with "a South Korean web artist" and quickly degenerates into incoherent text
- This is why sometimes ChatGPT gives you the same answer twice and sometimes it's different
  - The temperature setting determines how much randomness is in the sampling
- Most production systems use moderate temperature — enough creativity to be useful, not so much that it's unreliable

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

- This is the complete Transformer pipeline — it's surprisingly repetitive
  - The same pair of operations — attention then MLP — runs 96 times
  - Each pass refines the model's understanding
- Attention is where words communicate with each other
  - "Which other words are relevant to my meaning right now?"
  - This is how context gets incorporated — how "king" becomes "Scottish king in Macbeth"
  - We'll do a full deep-dive on attention in the next chapter
- MLP is where factual knowledge gets injected
  - Each word gets processed independently — no inter-word communication
  - "Is this vector encoding Michael Jordan? If so, add basketball, Chicago Bulls, #23"
  - About two-thirds of GPT-3's parameters live in the MLP layers
- After all 96 rounds, the final vector passes through unembedding and softmax
  - Produces a probability distribution over ~50,000 possible next tokens
  - The model samples one, appends it, and starts the whole pipeline again

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

- Let's see where all 175 billion parameters actually live
  - Embedding and unembedding: ~1.2 billion combined — less than 1% of the model
    - These are the "dictionary" layers that convert between words and vectors
  - Attention: ~58 billion — about one-third of the model
    - Query, key, value, and output matrices across 96 layers with 96 heads each
    - This is where context understanding happens
  - MLP: ~116 billion — about two-thirds of the model
    - Up-projection and down-projection matrices across 96 layers
    - This is where factual knowledge is stored
- The surprise: most of the model is MLP, not attention
  - Attention gets all the headlines, but the majority of parameters are in the knowledge-storage layers
- All 175 billion of these numbers were learned through gradient descent
  - The same algorithm we covered in Session 0 — just at mind-boggling scale

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

- Quick recap of the full pipeline
  - Text comes in, gets tokenized into chunks, each chunk becomes a vector of 12,288 numbers
  - Those vectors flow through 96 layers of attention (words talking to each other) and MLP (knowledge injection)
  - The final vector gets converted back to a probability distribution over ~50,000 possible next words
  - Temperature controls how much randomness goes into the selection
- Key numbers to remember:
  - 50,257 tokens in the vocabulary
  - 12,288 dimensions per embedding vector
  - 96 layers of attention + MLP
  - 175 billion total parameters
- Next up: we go deep on **attention** — the mechanism that lets words understand context
  - That's the heart of why Transformers work, and it's the subject of Chapter 7

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
