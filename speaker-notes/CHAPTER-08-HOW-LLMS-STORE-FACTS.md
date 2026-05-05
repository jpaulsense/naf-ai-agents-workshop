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

- Model reliably completes "Michael Jordan plays the sport of ___" → "basketball" across many phrasings
- Works for thousands of facts: capitals, dates, scientific knowledge — not luck
- Billions of parameters across many layers — the fact is encoded somewhere, but where?
- Active research question, not fully solved — but strong evidence points to a specific location

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

- DeepMind systematically tested which components matter — disrupting attention doesn't lose facts, disrupting MLPs does
- Division of labor: Attention = context/word relationships; MLP = knowledge bank/fact storage
- This gives us a specific place to look — now let's understand how MLP blocks work

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

- Tokens → embedding vectors → flow through alternating Attention (blue) and MLP (green) blocks
- Attention: words share context with each other; MLP: each word processed alone, knowledge injected
- GPT-3: 96 layers of each = 192 total blocks
- Final vector → word probabilities; today's focus = the green MLP blocks

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

- Four steps: up-projection (expand ~12K → ~50K), ReLU (kill negatives), down-projection (compress ~50K → ~12K), add back to original
- Expansion is ~4x the embedding dimension
- Up-projection = "asking questions," ReLU = "yes/no gate," down-projection = "injecting answers"

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

- ~50,000 rows (4x embedding dim), each row = a direction/detector in high-dimensional space
- Dot product = "how much does input align with this direction?" — positive = yes, negative = no
- All 50,000 questions asked simultaneously (massively parallel, why GPUs matter)
- Output: column of ~50,000 scores — still need bias + ReLU before it means anything

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

- One row encodes two directions: Michael + Jordan; dot product measures alignment with each
- Bias of -1 = threshold requiring both: "Michael Jordan" → 2-1=1 (fires); just "Michael" → 1-1=0 (doesn't)
- Functionally an AND gate — built from matrix multiply + bias, learned during training, not programmed
- Simple linear algebra creates logical operations

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

- ReLU: negative → zero, positive → unchanged; simplest useful function in AI
- Positive score = neuron "active" (feature detected); zero/negative = "inactive" (contributes nothing)
- With bias trick: "Michael Jordan" → ~1 → passes ReLU → ACTIVE; just "Michael" → ~0 → clipped → INACTIVE
- After ReLU: most of ~50,000 values are zero; only the few active neurons proceed to next step

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

- Down-projection: ~50,000 columns, each column = a direction in embedding space encoding what knowledge to add
- "Michael Jordan" neuron's column points in "basketball" direction — active → column added; inactive → zero contribution
- One column can encode multiple associated facts: basketball + Chicago Bulls + #23 + slam dunk
- Filing cabinet analogy: up-projection + ReLU = which folders to open; down-projection = the knowledge inside

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

- Full pipeline: "Michael Jordan" vector → up-projection (50K dot products) → one row detects "Michael+Jordan" (≈2) → bias -1 → ≈1 → ReLU passes → down-projection column = "basketball" direction → added to vector
- Output vector now carries "Michael Jordan + basketball" → unembedding gives "basketball" high probability
- No lookup table, no database, no if/then — knowledge is in the matrix weights, learned via gradient descent

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

- Each MLP block ≈ 1.2B params; 96 layers × 1.2B = ~116B params = two-thirds of GPT-3
- Attention gets the headlines but is only ~58B (one-third); MLPs are the quiet majority
- 116B numbers tuned by gradient descent on ~300B tokens of text
- Larger models know more facts: more MLP params = more knowledge storage capacity

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

- NOT one neuron = one concept; reality is superposition — features overlap, neurons participate in many concepts
- Like radio stations on overlapping frequencies — any single neuron encodes many things
- High-dimensional geometry: 2D → 2 perpendicular directions; 12,288D → rules change dramatically
- Johnson-Lindenstrauss: 100 dimensions can hold 10,000+ nearly-perpendicular vectors; growth is exponential
- Grant's quote: "10x dimensions = way more than 10x independent ideas" — why scaling works so disproportionately well

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

- Field = mechanistic interpretability — reverse-engineering what's inside neural networks
- Sparse autoencoders (Anthropic et al.): decompose superimposed neuron activations into individual interpretable features
- Analogy: multiple people talking → microphone picks up jumble → software separates individual voices
- Cutting-edge research, not fully solved yet — but critical for trust and safety

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
