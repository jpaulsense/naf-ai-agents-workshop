# Chapter 8 — How Might LLMs Store Facts

## Presentation Guide

**Duration:** 30–40 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build intuition for how large language models store and retrieve factual knowledge inside their MLP layers — understanding up-projection, ReLU gating, down-projection, and the mind-bending concept of superposition.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 7: "How might LLMs store facts"](https://www.youtube.com/watch?v=9-Jl0dxWQs8) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **The question: where do facts live?** | 3 min |
| 0:03 | **Google DeepMind's discovery** | 3 min |
| 0:06 | **Transformer recap** | 4 min |
| 0:10 | **The MLP computation** | 4 min |
| 0:14 | **Up-projection: asking questions** | 4 min |
| 0:18 | **The Michael Jordan example** | 4 min |
| 0:22 | **ReLU as gatekeeper** | 3 min |
| 0:25 | **Down-projection: injecting knowledge** | 4 min |
| 0:29 | **Putting it all together** | 3 min |
| 0:32 | **Parameter count and scale** | 3 min |
| 0:35 | **Superposition: packing more ideas than neurons** | 4 min |
| 0:39 | **Sparse autoencoders** | 2 min |

**Total: ~40 minutes (12 slides)**

---

## The Question: Where Do Facts Live?

### SLIDE 01 — The mystery of stored knowledge

**Headline:** "Michael Jordan plays the sport of ___" → basketball. But where does that fact live inside the network?

**Body:**
- An LLM can complete "Michael Jordan plays the sport of ___" with "basketball" — confidently and correctly
- The model has 175 billion parameters spread across dozens of layers
- The fact "Michael Jordan plays basketball" is encoded *somewhere* in those parameters
- But where? Which part of the network is responsible for storing and retrieving facts?

**Visual concept:** A prompt "Michael Jordan plays the sport of ___" with the word "basketball" appearing as the completion. Below it, a simplified Transformer diagram with a giant question mark over the middle layers.

**Speaker notes:**

- Start with the concrete question — type "Michael Jordan plays the sport of ___" and the model says "basketball"
  - Not a lucky guess — it does this reliably, across many phrasings
  - Same for thousands of other facts: capitals, dates, scientific knowledge
- The model has billions of parameters across many layers
  - The fact is in there *somewhere*
  - But which part? Is it in the attention layers? The embedding? Spread everywhere?
- This is a real active research question — not fully solved
  - But we have strong evidence pointing to a specific location
  - That's what this session is about

---

### SLIDE 02 — Google DeepMind's discovery

**Headline:** Facts appear to live in the MLP blocks — not in attention.

**Body:**
- Google DeepMind researchers investigated where factual knowledge is stored
- Finding: facts are primarily encoded in the **MLP (multi-layer perceptron)** blocks
- The **attention** blocks handle relationships between words — context, grammar, pronoun resolution
- The **MLP** blocks handle knowledge — facts, associations, learned information
- Two different jobs, two different parts of the architecture

**Visual concept:** A Transformer layer split into two halves: Attention (labeled "word relationships") and MLP (labeled "knowledge and facts"), with the MLP half highlighted and a "FACTS LIVE HERE" callout.

**Key insight card:**
Attention figures out *which words matter to each other*. MLPs figure out *what the model knows about those words*.

**Speaker notes:**

- Google DeepMind research — landmark finding about fact storage
  - Systematically tested which components matter for factual recall
  - Attention can be disrupted without losing facts
  - Disrupting MLP layers destroys factual knowledge
- Clean division of labor in the Transformer:
  - Attention = context, word relationships, "who is talking about whom"
  - MLP = knowledge bank, fact storage, "what do I know about this topic"
- This finding gives us a specific place to look
  - Let's understand how MLP blocks actually work

---

## Transformer Recap

### SLIDE 03 — The Transformer pipeline (refresher)

**Headline:** Tokens flow through alternating attention and MLP blocks — dozens of times.

**Body:**
- Text → **tokens** → **embeddings** (vectors of numbers)
- Embeddings flow through alternating blocks, repeated many times:
  - **Attention block:** Each word looks at every other word, updates its meaning based on context
  - **MLP block:** Each word processed independently — knowledge and facts get added
- Final vector → **unembedding** → probability distribution → next-word prediction

**Visual concept:** Vertical pipeline: tokens enter at bottom → embedding → [Attention → MLP] repeated with a "×96 layers" label → unembedding → prediction at top. Attention blocks in blue, MLP blocks in green.

**Speaker notes:**

- Quick refresher on the Transformer architecture
  - Tokens come in, get converted to embedding vectors (lists of numbers)
  - Those vectors flow upward through alternating blocks
- Two block types, alternating:
  - Attention (blue) — words talk to each other, share context
  - MLP (green) — each word processed alone, knowledge injected
- GPT-3: 96 layers of each = 192 total blocks
- After all layers, final vector gets converted back to word probabilities
- Today's focus: those green MLP blocks
  - What's actually happening inside them?
  - How do they store and retrieve facts?

---

## The MLP Computation

### SLIDE 04 — Inside the MLP: a four-step process

**Headline:** Each MLP block runs four operations on every vector that passes through.

**Body:**
1. **Up-projection:** Multiply the vector by a large matrix (+ add bias) — expands into ~50,000 values
2. **ReLU:** Clip all negative values to zero — only positive values survive
3. **Down-projection:** Multiply by a second matrix — compresses back to original size
4. **Add back:** Result gets added to the original vector (residual connection)

**Visual concept:** A horizontal flow diagram: Input vector (small) → UP-PROJECTION MATRIX (expands to tall column of ~50,000 values) → ReLU GATE (some values zeroed, some pass) → DOWN-PROJECTION MATRIX (compresses back to small) → ADD to original → Output vector

**Speaker notes:**

- Four steps — each one has a specific job
  - Step 1: Up-projection — expand the vector into a much larger space (~50,000 values)
  - Step 2: ReLU — kill all negative values, keep positives
  - Step 3: Down-projection — compress back down to original size
  - Step 4: Add result back to the original vector
- The "up then down" shape is key
  - Go from ~12,000 dimensions up to ~50,000, then back down to ~12,000
  - The expansion is roughly 4× the embedding dimension
- Each step has a specific purpose — let's walk through them one at a time
  - Up-projection = "asking questions"
  - ReLU = "yes/no gate"
  - Down-projection = "injecting answers"

---

## Up-Projection: Asking Questions

### SLIDE 05 — Each row of the matrix asks a question

**Headline:** The up-projection matrix has ~50,000 rows — each one is a "question" about the input vector.

**Body:**
- The up-projection matrix has ~50,000 rows, each the same length as the embedding vector
- Each row is a **direction** in embedding space
- Computing the dot product of a row with the input = asking "does this vector align with this direction?"
- High dot product = "yes, strong match"
- Low or negative dot product = "no, not a match"
- ~50,000 rows = ~50,000 questions asked simultaneously

**Visual concept:** A matrix with many rows, each row labeled as a question: "Is this about basketball?", "Is this a person's name?", "Is this related to France?", etc. Arrows showing dot products between each row and the input vector, producing a column of scores.

**Analogy card:**
Think of it like a checklist with 50,000 yes/no questions. The vector walks down the checklist, and each question gets a score based on how well the vector matches that particular topic or feature.

**Speaker notes:**

- The matrix has ~50,000 rows — that's 4× the embedding dimension
  - Each row is a direction in the high-dimensional space
  - Think of each row as encoding a specific question or detector
- Dot product = "how much does my input align with this direction?"
  - High positive number = strong alignment = "yes"
  - Near zero = no particular alignment
  - Negative = opposite direction
- All 50,000 questions get asked simultaneously
  - Massively parallel — this is why GPUs matter
- After this step, you have a column of ~50,000 scores
  - Each score = the answer to one question about the input
  - But we're not done — the bias and ReLU haven't happened yet

---

### SLIDE 06 — The Michael Jordan example: an AND gate

**Headline:** One row might encode "Michael + Jordan" — and the bias makes it fire only when BOTH names are present.

**Body:**
- Imagine one row in the matrix points in the "Michael" direction AND the "Jordan" direction
- If the input vector encodes the full name "Michael Jordan":
  - Dot product with "Michael" component ≈ 1
  - Dot product with "Jordan" component ≈ 1
  - Total ≈ 2
- If only "Michael" is present: total ≈ 1
- If only "Jordan" is present: total ≈ 1
- **Bias of -1** is added: result is positive (≈1) ONLY for the full name
  - "Michael Jordan" → 2 - 1 = **1** (positive → fires)
  - Just "Michael" → 1 - 1 = **0** (zero → doesn't fire)
  - Just "Jordan" → 1 - 1 = **0** (zero → doesn't fire)

**Visual concept:** A row vector shown as two arrows: one pointing in "Michael" direction, one in "Jordan" direction. Three scenarios showing the dot product calculation for each case, with the bias subtraction making only the full-name case positive.

**Key insight card:**
The combination of directional weights + bias creates a logical AND gate: the neuron activates only when BOTH conditions are met. Simple arithmetic producing sophisticated logic.

**Speaker notes:**

- Walk through the Michael Jordan example step by step
  - One row in the up-projection matrix encodes two directions: Michael + Jordan
  - Dot product measures alignment with EACH direction
- The crucial trick is the bias
  - Bias of -1 means the total must exceed 1 to produce a positive result
  - Full name "Michael Jordan" → ~2, minus 1 = ~1 → positive
  - Just "Michael" → ~1, minus 1 = ~0 → not positive
  - Just "Jordan" → ~1, minus 1 = ~0 → not positive
- This is functionally an AND gate
  - Fires if Michael AND Jordan, not just one or the other
  - Built entirely from matrix multiplication and a bias term
  - No one programmed this logic — the network learned it during training
- Powerful concept: simple linear algebra creates logical operations

---

## ReLU as Gatekeeper

### SLIDE 07 — ReLU: the simplest possible gate

**Headline:** Negative → zero. Positive → pass through. That's the entire rule.

**Body:**
- **ReLU** (Rectified Linear Unit): the activation function used in modern networks
- Rule: if the value is negative, set it to zero. If positive, let it through unchanged.
- After up-projection + bias, each of the ~50,000 values hits ReLU
- **Active neuron** = positive value survived → this feature was detected
- **Inactive neuron** = value was zero or negative → clipped to zero → this feature was NOT detected
- Combined with the bias trick from the previous slide, ReLU enforces the AND gate

**Visual concept:** A simple graph showing the ReLU function: flat at zero for all negative inputs, then a straight diagonal line for positive inputs. Below: a column of 50,000 values, some highlighted green (positive, survived) and others grayed out (zeroed by ReLU).

**Speaker notes:**

- ReLU is dead simple — possibly the simplest useful function in all of AI
  - Negative input → output is zero
  - Positive input → output equals the input, unchanged
  - That's it — no fancy math
- Why it matters here:
  - After up-projection + bias, each "question" has a score
  - ReLU decides: did the score survive? Is it positive?
  - If yes → this neuron is "active" → this feature was detected
  - If no → this neuron is "inactive" → zeroed out → contributes nothing
- Combined with the bias trick:
  - "Michael Jordan" → score of ~1 → ReLU passes it through → ACTIVE
  - Just "Michael" → score of ~0 → ReLU clips to zero → INACTIVE
  - The AND gate is now fully operational
- After ReLU, you have ~50,000 values
  - Most are zero (inactive) — the input didn't match those features
  - A few are positive (active) — these features were detected
  - Only the active neurons move on to the next step

---

## Down-Projection: Injecting Knowledge

### SLIDE 08 — Each column encodes a fact to inject

**Headline:** Think column by column. Each active neuron's column = a direction in embedding space — the knowledge to add.

**Body:**
- The down-projection matrix has ~50,000 columns (one per neuron from the up-projection)
- Each column is a **direction in embedding space** — it encodes specific information
- Example: the column for the "Michael Jordan" neuron might point in the "basketball" direction
- **Active neuron** → its column gets added (scaled by the activation value)
- **Inactive neuron** → its column is multiplied by zero → contributes nothing
- A single column can encode **multiple** associated facts — basketball AND Chicago Bulls AND #23

**Visual concept:** A matrix shown column by column. One column highlighted, labeled "basketball / Chicago Bulls / #23 direction." An active neuron (value = 1) multiplies this column, producing the full vector. An inactive neuron (value = 0) produces a zero vector. The active column gets added to the original embedding.

**Analogy card:**
Think of a filing cabinet with 50,000 folders. Each folder contains knowledge about a specific topic. The up-projection + ReLU decide which folders to open. The down-projection is the content inside those folders, which gets added to the model's working knowledge.

**Speaker notes:**

- Now think about the down-projection matrix column by column
  - Each of the ~50,000 columns corresponds to one neuron
  - Each column IS a direction in embedding space — it encodes what to add
- The "Michael Jordan" neuron's column:
  - Points in the "basketball" direction in embedding space
  - When this neuron is active, the column gets scaled by the activation value and added
  - When inactive (zeroed by ReLU), the column is multiplied by zero — contributes nothing
- Single column, multiple facts:
  - One column can encode "basketball + Chicago Bulls + jersey #23 + slam dunk"
  - All packed into one direction in the high-dimensional space
  - Directions can carry multiple pieces of associated information simultaneously
- Filing cabinet analogy:
  - 50,000 folders, each containing knowledge about a specific topic
  - Up-projection + ReLU = deciding which folders to open
  - Down-projection = the actual knowledge inside the folder
  - Only opened folders contribute to the output

---

## Putting It All Together

### SLIDE 09 — The full MLP pipeline for one fact

**Headline:** Vector encodes "Michael Jordan" → AND gate fires → "basketball" direction added → output knows the answer.

**Body:**
- **Input:** Vector flowing through the network currently encodes "Michael Jordan"
- **Up-projection:** 50,000 dot products — one row detects "Michael + Jordan" pattern
- **Bias:** Threshold ensures both names must be present (AND gate)
- **ReLU:** Neuron activates (positive value survives)
- **Down-projection:** Active neuron's column = "basketball" direction → gets added to the vector
- **Output:** Vector now encodes "Michael Jordan + basketball" → model is ready to predict "basketball" as the next word

**Visual concept:** A single horizontal pipeline showing the full flow: "Michael Jordan" vector → matrix multiply → bias subtract → ReLU gate (green = pass) → column lookup → add "basketball" direction → enriched output vector. Each step labeled with the concrete values from the example.

**Key insight card:**
The entire process is just matrix multiplication, subtraction, and zeroing out negatives. Simple operations, repeated at enormous scale, producing what looks like "knowing facts."

**Speaker notes:**

- Walk through the complete pipeline end to end
  - Start: vector that encodes "Michael Jordan" arrives at an MLP block
  - Up-projection: 50,000 questions asked simultaneously via dot products
  - One row = the "Michael + Jordan" detector → dot product ≈ 2
  - Bias of -1 → result ≈ 1 → positive
  - ReLU: positive value passes through → neuron is ACTIVE
  - Down-projection: active neuron's column = "basketball" direction
  - Scaled by activation value and added to original vector
  - Output: vector now carries both "Michael Jordan" AND "basketball"
- The final unembedding layer later converts this enriched vector into probabilities
  - "Basketball" gets a high probability as the next word
- All from matrix math — no lookup table, no database, no if/then rules
  - The "knowledge" is encoded in the weights of two matrices
  - Learned entirely from training data via gradient descent

---

## Parameter Count and Scale

### SLIDE 10 — Two-thirds of the model is fact storage

**Headline:** Each MLP block: ~1.2 billion parameters. 96 layers × 1.2B = ~116 billion. That's two-thirds of GPT-3.

**Body:**
- Each MLP block has two large matrices (up-projection + down-projection) plus biases
- Per block: approximately **1.2 billion parameters**
- GPT-3 has **96 layers**, each with an MLP block
- 96 × 1.2B = **~116 billion parameters** devoted to MLPs
- GPT-3 total: 175 billion parameters
- MLP share: **~66%** — two-thirds of the entire model is knowledge storage

| Component | Parameters | Share |
|-----------|-----------|-------|
| Embeddings + unembedding | ~1.2 billion | <1% |
| Attention (96 layers) | ~58 billion | ~33% |
| **MLP (96 layers)** | **~116 billion** | **~66%** |
| **Total** | **~175 billion** | **100%** |

**Speaker notes:**

- The scale of MLP layers is staggering
  - Each MLP block: two matrices + biases ≈ 1.2 billion parameters
  - 96 layers × 1.2B = ~116 billion parameters
  - That's about two-thirds of all of GPT-3
- Attention gets all the headlines — "Attention Is All You Need"
  - But attention is only one-third of the parameters (~58 billion)
  - The quiet majority of the model is MLP fact storage
- Think about what 116 billion parameters means:
  - 116 billion individual numbers, each tuned by gradient descent
  - Each one contributing to the model's ability to store and retrieve facts
  - Trained on ~300 billion tokens of text
- This is why larger models know more facts
  - More MLP parameters = more capacity for knowledge storage
  - More training data = more facts to encode

---

## Superposition: Packing More Ideas Than Neurons

### SLIDE 11 — Nearly perpendicular directions in high-dimensional space

**Headline:** Individual neurons rarely represent single clean features. The model packs far more concepts than it has neurons — using geometry.

**Body:**
- You might expect: one neuron = one concept (the "Michael Jordan neuron")
- Reality: concepts are spread across many neurons, and neurons participate in many concepts
- This is called **superposition** — features overlap and share neurons
- Why it works: in high-dimensional space, you can have MANY nearly perpendicular directions
- **Johnson-Lindenstrauss lemma:** 100 dimensions can hold **10,000+** nearly perpendicular vectors
- Grant's quote: *"A space with 10× as many dimensions can store way, way more than 10× as many independent ideas."*
- The growth is **exponential** — doubling dimensions doesn't just double capacity, it multiplies it enormously

**Visual concept:** A 2D diagram showing two perpendicular arrows (only 2 independent directions possible). Next to it, a 3D diagram showing three. Then a "12,288-D" label with thousands of arrows radiating outward, labeled "billions of nearly-perpendicular directions." Include a scaling curve showing exponential growth of capacity vs. dimensions.

**Key insight card:**
This is why bigger models are dramatically more capable — not just proportionally better. Doubling the dimensions gives exponentially more room for knowledge. It's not linear, it's explosive.

**Speaker notes:**

- The naive expectation: each neuron = one clean concept
  - "The Michael Jordan neuron," "the basketball neuron," etc.
  - This is NOT how it works in practice
- Reality: **superposition**
  - Features overlap and share neurons
  - Like multiple radio stations broadcasting on overlapping frequencies
  - Any single neuron participates in encoding many different concepts
- Why this works — high-dimensional geometry:
  - In 2D, you can only have 2 perpendicular directions
  - In 3D, you can have 3
  - But in 12,288 dimensions? The rules change dramatically
- Johnson-Lindenstrauss lemma:
  - 100 dimensions can hold 10,000+ nearly perpendicular vectors
  - "Nearly perpendicular" = close enough to independent that they don't interfere much
  - The growth is exponential, not linear
- Grant's key quote: "A space with 10 times as many dimensions can store way, way more than 10 times as many independent ideas"
  - This is why scaling models up makes them disproportionately more capable
  - Double the parameters → exponentially more room for knowledge

---

### SLIDE 12 — Sparse autoencoders: extracting the true features

**Headline:** Researchers are building tools to untangle the superimposed features and see what the model actually learned.

**Body:**
- Since neurons don't represent clean individual concepts, how do you figure out what the model actually learned?
- **Sparse autoencoders** — a technique developed by Anthropic (the makers of Claude) and others
- They take the messy, superimposed neuron activations and decompose them into interpretable features
- Think of it as: the model learned thousands of clean concepts, but encoded them as overlapping combinations across neurons
- Sparse autoencoders reverse-engineer those clean concepts from the overlapping signals
- Active research area — we're still learning how to read what's inside these models

**Visual concept:** A tangled ball of overlapping colored threads (representing superimposed features in neurons) → a "Sparse Autoencoder" box → individual separated threads, each labeled with a clean concept: "basketball," "French cuisine," "historical dates," etc.

**Speaker notes:**

- If neurons don't cleanly represent concepts, how do we understand what the model learned?
  - This is the field of **mechanistic interpretability**
  - Trying to reverse-engineer what's going on inside neural networks
- Sparse autoencoders are one key tool:
  - Developed significantly by Anthropic (makers of Claude)
  - Take the overlapping, superimposed neuron activations
  - Decompose them into individual, interpretable features
- Analogy: multiple people talking at once in a room
  - A single microphone picks up a jumbled mix of voices
  - Sparse autoencoder = software that separates the individual voices
  - Each separated voice = one clean feature the model learned
- This is cutting-edge research — still an active area
  - We don't yet have complete tools to fully understand what's inside these models
  - But progress is accelerating
- Why it matters: understanding what models know (and don't know) is critical for trust and safety

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| The question | 1 (slide 01) | 3 min |
| DeepMind's discovery | 1 (slide 02) | 3 min |
| Transformer recap | 1 (slide 03) | 4 min |
| The MLP computation | 1 (slide 04) | 4 min |
| Up-projection | 2 (slides 05-06) | 8 min |
| ReLU gatekeeper | 1 (slide 07) | 3 min |
| Down-projection | 1 (slide 08) | 4 min |
| Full pipeline | 1 (slide 09) | 3 min |
| Parameter count | 1 (slide 10) | 3 min |
| Superposition | 1 (slide 11) | 4 min |
| Sparse autoencoders | 1 (slide 12) | 2 min |
| **Total** | **12 slides** | **~40 minutes** |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 7: ["How might LLMs store facts"](https://www.youtube.com/watch?v=9-Jl0dxWQs8) (2024)

Additional references:
- Google DeepMind research on fact localization in Transformer MLP layers
- Anthropic's work on sparse autoencoders and mechanistic interpretability
