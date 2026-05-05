# Presenter Preparation Guide

**What this is:** A guided self-check you can walk through before presenting. For each topic, try to hit the key beats from memory. If you blank on something, go back to the chapter notes and review, then try again later. The goal is fluency, not perfection — you want every concept to feel natural when you're standing in front of the room.

**How to use it:** Go section by section. For each prompt, talk through your answer out loud (or in your head). Check the "beats to hit" underneath — did you cover them? Anything you missed, flag it and come back next round.

---

## Chapter 1 — What Is a Neural Network?

### The Hook

**Prompt:** Why is handwritten digit recognition the perfect opening example?

Beats to hit:
- [ ] Humans do it instantly, but can't describe how ("What counts as a curve? How curvy?")
- [ ] Every rule you write has a hundred exceptions
- [ ] Core insight of ML: don't tell the computer the rules, show it examples
- [ ] 28x28 pixel images — the "hello world" of AI
- [ ] Simple enough to understand, contains every concept that scales to ChatGPT

### Neurons

**Prompt:** What is a neuron? Strip away all the mystique.

Beats to hit:
- [ ] Just a number between 0 and 1
- [ ] NOT a brain cell — a container holding a value
- [ ] The value is called its **activation**
- [ ] Dimmer switch analogy (not on/off, it's a range)
- [ ] Everything AI does comes from networks of these interacting

### Layers

**Prompt:** Walk through the 4-layer structure for digit recognition.

Beats to hit:
- [ ] **Input layer**: 784 neurons (one per pixel, 28x28). Each holds pixel brightness 0-1.
- [ ] **Hidden layer 1**: 16 neurons — detects edges, small shapes
- [ ] **Hidden layer 2**: 16 neurons — combines into patterns, loops
- [ ] **Output layer**: 10 neurons (one per digit 0-9). Brightest = answer.
- [ ] Hierarchical decomposition — each layer handles one level of abstraction
- [ ] Digit examples: 9 = loop on top + line on right. 4 = three lines.
- [ ] Military analogy: front line collects raw intel, middle echelons analyze, top makes the call

### Weights and Bias

**Prompt:** Explain weights and bias to someone who's never heard of them.

Beats to hit:
- [ ] **Weight** = number on each connection (volume knob controlling influence)
- [ ] Positive weight: "if you light up, I should too"
- [ ] Negative weight: "if you light up, I should stay quiet"
- [ ] Size of weight = strength of influence
- [ ] Edge detection example: positive weights where edge should be, negative on surrounding pixels
- [ ] Total connections: 784 x 16 = 12,544 in first layer alone
- [ ] **13,002 total adjustable parameters** — that IS the network
- [ ] **Bias** = threshold for "how strong must the signal be before I care?"

### The 4-Step Computation

**Prompt:** What does every single neuron do?

Beats to hit:
- [ ] 1. **Multiply** each input activation by its weight
- [ ] 2. **Add** them all up (weighted sum)
- [ ] 3. **Add a bias** (threshold)
- [ ] 4. **Squish** into 0-1 range (activation function)
- [ ] **Sigmoid**: smoothly squishes any number into 0-1. Grant calls it "sigmoid squishification."
- [ ] **ReLU**: negative → 0, positive → pass through. Modern networks use this because sigmoid flattens at extremes.
- [ ] This one operation, repeated thousands of times, produces all complex AI behavior

### The Big Demystification

**Prompt:** What's the one-sentence version of what a neural network is?

Beats to hit:
- [ ] A mathematical function: 784 numbers in → 10 numbers out
- [ ] Can be written as **sigma(W dot a + b)**
- [ ] Random parameters = garbage. Right parameters = 98% accuracy.
- [ ] Humans design the architecture. The machine learns the parameter values.
- [ ] Nothing mystical — a function with a lot of knobs

---

## Chapter 2 — Gradient Descent (How Networks Learn)

### Cost Function

**Prompt:** Before the network can improve, what does it need first?

Beats to hit:
- [ ] A way to measure how wrong it is — the **cost function**
- [ ] Show it a 3, the "3" neuron should be high, everything else low
- [ ] Take difference between actual and desired output, square each, add them up
- [ ] Average across all training examples = overall "grade"
- [ ] Low cost = good. High cost = confused.
- [ ] Inverted exam score analogy: zero is perfect

### The Algorithm

**Prompt:** Explain gradient descent like you're describing a blindfolded person on a hillside.

Beats to hit:
- [ ] Cost function as a landscape — elevation = how wrong
- [ ] Start at random position (random parameters = high elevation)
- [ ] Blindfolded — can only feel slope under your feet
- [ ] Compute gradient (direction of steepest uphill), go opposite direction
- [ ] Take a small step downhill. Recompute. Repeat.
- [ ] **No guarantee** you find the deepest valley — might land in a local minimum
- [ ] Step size proportional to slope — naturally smaller steps near the bottom, avoids overshooting
- [ ] This is the **single most important algorithm in modern AI**

### The Gradient Vector

**Prompt:** What does the gradient actually tell you? Make it concrete.

Beats to hit:
- [ ] A list of 13,002 numbers — one per parameter
- [ ] **Sign** (positive/negative): should this parameter go up or down?
- [ ] **Magnitude** (size): how sensitive is the cost to this parameter?
- [ ] Gradient component of 3.2 vs. 0.1 → cost is 32x more sensitive to the first
- [ ] It's a ranked priority list: which knobs to turn, how much, in which direction
- [ ] "Encodes the relative importance of each weight and bias"

### Stochastic Gradient Descent

**Prompt:** Why don't we process all training data at once?

Beats to hit:
- [ ] Too expensive — tens of thousands of examples per step
- [ ] **Mini-batches** of ~100 random examples
- [ ] Each mini-batch gives a pretty good approximation of the true gradient
- [ ] "Drunk man stumbling down a hill taking quick steps" vs. "careful man taking slow precise steps"
- [ ] Noisy but fast — overall trend is still downhill
- [ ] Polling analogy: you don't survey every person in the country

---

## Chapter 3 — Backpropagation

### What It Does

**Prompt:** What is backpropagation in one sentence, then unpack it.

Beats to hit:
- [ ] The algorithm that computes the gradient efficiently
- [ ] Name literally describes it: propagate errors backward through the network
- [ ] For each training example, figures out how each weight and bias should change

### Walking Through One Example

**Prompt:** Walk through backprop with a single image of a "2."

Beats to hit:
- [ ] Output is wrong — "2" neuron is too low, "8" neuron is too high
- [ ] Start at output, work backward
- [ ] Three ways to help increase the "2" neuron:
  - [ ] 1. Increase the **bias**
  - [ ] 2. Increase the **weights** (especially on connections with bright/active neurons)
  - [ ] 3. Change **previous layer's activations** (desired activations propagated backward)
- [ ] Adjustments proportional to activations — bright neurons = more leverage
- [ ] **Hebbian theory**: "neurons that fire together wire together"
- [ ] But the "2" neuron isn't the only one with opinions — ALL output neurons want changes
- [ ] Add all desires together, propagate backward, repeat layer by layer

### The Averaging Problem

**Prompt:** Why can't you just use one training example?

Beats to hit:
- [ ] One image of a "2" creates adjustments biased toward that specific example
- [ ] Would incentivize classifying everything as a 2
- [ ] Need to average desired changes across many examples
- [ ] That average = the negative gradient (or proportional to it)

---

## Chapter 4 — Backpropagation Calculus

### The Chain Rule

**Prompt:** How do you trace how a tiny change in one weight affects the final cost?

Beats to hit:
- [ ] Tiny nudge to weight w → nudges weighted sum z → nudges activation a → nudges cost C
- [ ] Multiply the ratios: (dC/da) x (da/dz) x (dz/dw)
- [ ] That's the **chain rule** — the math behind backpropagation
- [ ] Three key pieces:
  - [ ] dC/da = 2(a - y) — bigger error = bigger sensitivity
  - [ ] da/dz = derivative of activation function (sigmoid or ReLU)
  - [ ] dz/dw = previous neuron's activation — connects back to Hebbian learning
- [ ] Sensitivity to bias is almost identical (dz/db = 1)
- [ ] To propagate backward: sensitivity to previous activation = weight w(L)
- [ ] Multiple neurons per layer: sum over all paths the neuron influences

---

## Chapter 5 — Large Language Models (Brief Overview)

### The Core Idea

**Prompt:** What is an LLM at its most basic?

Beats to hit:
- [ ] A mathematical function that predicts the next word
- [ ] Assigns probability to every possible next word (~50,000 options)
- [ ] Movie script analogy: AI response is torn off, prediction machine fills it in word by word
- [ ] To build a chatbot: set up interaction format, have model predict assistant's response
- [ ] Allow less likely words sometimes → more natural output
- [ ] Model is deterministic, but random sampling → different answers each time

### Training and Scale

**Prompt:** How does an LLM learn, and how big is the computation?

Beats to hit:
- [ ] Parameters start random (gibberish), refined by comparing predictions to actual next words
- [ ] **Backpropagation** tweaks all parameters to make correct prediction more likely
- [ ] Scale: "a billion operations per second, it would take over 100 million years"
- [ ] **Pre-training**: auto-completing internet text (2,600+ years of reading for a human)
- [ ] **RLHF** (Reinforcement Learning with Human Feedback): workers flag unhelpful predictions, further refine
- [ ] GPUs: special chips optimized for parallel operations — essential for this scale

### Transformers

**Prompt:** What makes transformers different from earlier language models?

Beats to hit:
- [ ] Pre-2017: models processed text one word at a time (sequential)
- [ ] Transformers: soak it all in at once, in parallel
- [ ] Introduced in 2017 by Google ("Attention Is All You Need")
- [ ] Two key operations: **attention** (words refine each other's meaning from context) and **feed-forward** (stores patterns/knowledge)
- [ ] Specific behavior is **emergent** from parameter tuning — not explicitly programmed

---

## Chapter 6 — Transformers (Architecture Deep Dive)

### The Data Pipeline

**Prompt:** Walk through what happens to text from the moment it enters a transformer to the prediction.

Beats to hit:
- [ ] Text → **tokens** (words or word pieces). Vocabulary: ~50,257.
- [ ] Each token → **embedding vector** (12,288 numbers for GPT-3). Looked up from **embedding matrix** (W_E, ~617M parameters).
- [ ] Vectors flow through alternating **attention blocks** and **MLP blocks** — 96 times for GPT-3
- [ ] Final vector → **unembedding matrix** (W_U, ~617M parameters) → 50,000 scores
- [ ] Scores → **softmax** → probability distribution → sample next word
- [ ] Total embedding + unembedding: ~1.2 billion parameters (less than 1% of total)

### Embeddings and Meaning

**Prompt:** Why is the embedding space so fascinating? Give the classic examples.

Beats to hit:
- [ ] Words with similar meanings land nearby in 12,288-dimensional space
- [ ] Woman - Man ≈ Queen - King (gender direction)
- [ ] Italy - Germany + Hitler ≈ Mussolini (nationality + historical role as separate directions)
- [ ] Germany - Japan + Sushi ≈ Bratwurst
- [ ] Cats - Cat = plurality direction. Dot product with "one," "two," "three," "four" → increasing values
- [ ] Nobody programmed any of this — structure emerges from training data
- [ ] **Dot product** measures alignment: positive = similar direction, zero = unrelated, negative = opposite

### Temperature and Softmax

**Prompt:** What is temperature, and how does it affect output?

Beats to hit:
- [ ] **Softmax**: turns raw scores into valid probability distribution (all positive, sums to 1)
- [ ] **Temperature** controls randomness of sampling
- [ ] T = 0: always pick most likely word (deterministic, repetitive)
- [ ] High T: more creative, but risks nonsense
- [ ] "Once upon a time" example: T=0 → trite Goldilocks story. High T → starts original, degenerates.
- [ ] Raw scores before softmax are called **logits**

---

## Chapter 7 — Attention (The Key Innovation)

### Why Attention Exists

**Prompt:** What problem does attention solve? Give the examples.

Beats to hit:
- [ ] Same word, different meanings depending on context
- [ ] "Mole": American shrew **mole** (animal) / one **mole** of CO2 (chemistry) / biopsy of the **mole** (skin growth)
- [ ] "Tower": Eiffel tower (Paris, iron, 1000 ft) vs. miniature tower (tiny)
- [ ] "Harry": wizard in passage → Harry Potter. Queen + Sussex → Prince Harry.
- [ ] Initial embeddings are identical regardless of context — just a lookup table
- [ ] Attention = mechanism for surrounding words to update each other's meanings

### Query, Key, Value

**Prompt:** Walk through a single attention head with "a fluffy blue creature roamed the verdant forest."

Beats to hit:
- [ ] **Query** (Q): what am I looking for? Creature asks: "any adjectives near me?"
  - Query matrix transforms embeddings into smaller query space (128 dimensions)
- [ ] **Key** (K): what do I have to offer? Fluffy says: "I'm a descriptor!"
  - Key matrix transforms embeddings into same small space
- [ ] Match via **dot product** between each query-key pair
  - High dot product = relevant. Low/negative = unrelated.
- [ ] **Softmax** normalizes columns → **attention pattern** (weights summing to 1)
- [ ] **Masking**: later words can't attend to future words (set to -infinity before softmax)
- [ ] **Value** (V): the actual information to add. Fluffy's value = the semantic content of fluffiness.
  - Weighted sum by attention weights → added to creature's embedding
  - Result: "creature" now encodes "fluffy blue creature"

### Multi-Head Attention

**Prompt:** Why 96 heads? What does that buy you?

Beats to hit:
- [ ] Each head has its own Q, K, V matrices — learns a different type of relationship
- [ ] One head might learn grammar (subject → verb)
- [ ] Another learns pronoun references ("it" → "animal" not "street")
- [ ] Another disambiguates (wizard context → Harry Potter; Queen/Sussex → Prince Harry)
- [ ] GPT-3: 96 heads per layer x 96 layers = **9,216 total attention operations**
- [ ] All 96 heads produce proposed changes, summed together → added to embedding
- [ ] ~58 billion parameters devoted to attention = **1/3 of GPT-3**

---

## Chapter 8 — How LLMs Store Facts (MLP Layers)

### The MLP Computation

**Prompt:** Walk through how the network stores and retrieves the fact "Michael Jordan plays basketball."

Beats to hit:
- [ ] **Step 1 — Up-projection**: multiply embedding by huge matrix (~50,000 rows). Each row asks a question via dot product.
  - Row = "Michael + Jordan direction." Dot product = 2 if full name, ≤1 otherwise.
  - Bias of -1: positive ONLY for the full name. **Acts like an AND gate.**
- [ ] **Step 2 — ReLU**: negative → 0, positive → unchanged. Neuron is "active" or "inactive."
  - Combined with bias: fires for "Michael Jordan" but NOT for just "Michael" or just "Jordan"
- [ ] **Step 3 — Down-projection**: think column by column. Each column = a direction (e.g., "basketball").
  - Active neuron → its column gets added. Inactive → no effect.
  - Single column can encode multiple facts: basketball + Chicago Bulls + #23
- [ ] **Step 4**: result added back to original embedding. Now encodes "Michael Jordan + basketball."

### Scale

**Prompt:** How much of the model is MLP?

Beats to hit:
- [ ] Each MLP block: ~1.2 billion parameters
- [ ] 96 layers → ~**116 billion parameters** total in MLPs
- [ ] That's **2/3 of GPT-3's 175 billion**
- [ ] Attention gets the headlines, but majority of parameters = knowledge storage

### Superposition

**Prompt:** Do individual neurons represent single clean concepts?

Beats to hit:
- [ ] Rarely. Neurons don't cleanly map to single features like "Michael Jordan"
- [ ] **Superposition**: features overlap, share neurons — like multiple radio stations on overlapping frequencies
- [ ] **Johnson-Lindenstrauss lemma**: in high-dimensional space, you can pack exponentially more nearly-perpendicular directions than dimensions
- [ ] 100 dimensions → 10,000+ nearly perpendicular vectors (89-91 degrees apart)
- [ ] "A space with 10x as many dimensions can store way, way more than 10x as many independent ideas"
- [ ] This may explain why bigger models are dramatically more capable
- [ ] **Sparse autoencoders**: tools for extracting true features from superimposed neurons (Anthropic research)

---

## Chapter 9 — AI Images and Video (Diffusion Models)

### The Core Process

**Prompt:** How does a diffusion model generate a video from nothing?

Beats to hit:
- [ ] Starts with **pure noise** (random pixel values)
- [ ] Passes through a **transformer** (same architecture as LLMs, but outputs video not text)
- [ ] Output: slightly less noisy video. Add back to noise, pass through again.
- [ ] Repeat ~50 times. Step by step, noise → realistic video.
- [ ] Connected to physics: **Brownian motion** run in reverse

### CLIP — Connecting Words and Images

**Prompt:** How does the model know what you're asking for in a text prompt?

Beats to hit:
- [ ] **CLIP** = two models (text encoder + image encoder), trained on 400M image-caption pairs
- [ ] Output: 512-dimensional vectors in a shared space
- [ ] **Contrastive training**: matching pairs align, non-matching pairs pushed apart
- [ ] Result: mathematical operations on pure concepts work — hat/no-hat difference vector → closest text is "hat"
- [ ] BUT: CLIP only goes one direction (text/image → embedding). Can't generate from embeddings.

### The Diffusion Process

**Prompt:** Walk through the training and generation of a diffusion model.

Beats to hit:
- [ ] **Training**: take clean images, add noise step by step until destroyed. Train model to predict **total noise added** (not single step).
- [ ] 2D spiral intuition: images as points in high-dimensional space. Adding noise = random walks.
- [ ] Model learns a **vector field** pointing back toward original data distribution
- [ ] **Time conditioning**: essential. Large t = coarse structure. Small t = fine detail. Phase-change behavior.
- [ ] **Why noise during generation matters**: without it, all points converge to the mean (blurry images)
- [ ] Model actually learns the **mean** of the distribution → need random noise to sample the full distribution

### Guidance — Steering with Text

**Prompt:** How does classifier-free guidance work?

Beats to hit:
- [ ] Compare **conditioned** model output (with text prompt) vs. **unconditioned** (no text)
- [ ] Subtract unconditioned direction from conditioned direction
- [ ] Amplify the difference with scaling factor **alpha**
- [ ] Higher alpha → stronger prompt adherence. "Tree literally grows in size and detail."
- [ ] **Negative prompts**: WAN model writes out unwanted features ("extra fingers, walking backwards"), subtracts that vector
- [ ] **DDIM**: deterministic alternative using ordinary differential equations. Fewer steps, same quality distribution.

---

## Key Numbers to Recall

Run through these from memory. If you can rattle them off, you're ready.

| Stat | Value |
|------|-------|
| Digit network input | 28x28 = 784 pixels |
| Digit network hidden layers | 2 layers, 16 neurons each |
| Digit network total parameters | 13,002 |
| GPT-3 total parameters | 175 billion |
| GPT-3 training tokens | ~300 billion |
| GPT-3 embedding dimension | 12,288 |
| GPT-3 vocabulary size | 50,257 tokens |
| GPT-3 context window | 2,048 tokens |
| GPT-3 layers | 96 |
| GPT-3 attention heads per layer | 96 |
| Total attention operations | 96 x 96 = 9,216 |
| Attention parameters | ~58 billion (1/3 of model) |
| MLP parameters | ~116 billion (2/3 of model) |
| Embedding + unembedding | ~1.2 billion |
| MLP neurons per block | ~50,000 (4x embedding dim) |
| CLIP vector dimension | 512 |
| CLIP training pairs | 400 million |
| Query/key space dimension | 128 |

---

## Quick-Fire Analogies

Can you deliver each of these naturally, as if it just occurred to you?

- [ ] **Neuron** → dimmer switch, not an on/off toggle
- [ ] **Layers** → military chain of command (raw intel → analysis → decision)
- [ ] **Weights** → volume knobs
- [ ] **Cost function** → inverted exam score (zero is perfect)
- [ ] **Gradient descent** → blindfolded person on a hillside, feeling for downhill
- [ ] **Gradient vector** → ranked priority list of which knobs to turn
- [ ] **Stochastic gradient descent** → drunk man stumbling downhill taking quick steps
- [ ] **Backpropagation** → after-action review (trace blame backward through chain of command)
- [ ] **Embeddings** → directions = concepts. Nearby = similar meaning.
- [ ] **Attention** → networking event (name tag = key, lanyard = query, best conversations = highest dot products)
- [ ] **MLP up-projection** → asking 50,000 yes/no questions simultaneously
- [ ] **MLP down-projection** → injecting the answers as knowledge
- [ ] **Superposition** → multiple radio stations broadcasting on overlapping frequencies
- [ ] **Diffusion** → Brownian motion run backward

---

## Presentation Flow Check

Can you walk through the entire 2-hour arc in under 3 minutes? Hit these beats:

1. [ ] AI is a function — inputs in, outputs out, machine learns the rules from examples
2. [ ] Neurons are just numbers. Stack them in layers. Connect with weights.
3. [ ] Each neuron does one thing: weighted sum, add bias, squish. Repeat 13,000 times.
4. [ ] Cost function measures how wrong. Gradient descent rolls downhill. Backprop computes the gradient.
5. [ ] Scale: mini-batches, stochastic gradient descent. Same algorithm, 175 billion parameters, 300 billion training tokens.
6. [ ] Shift to language: tokens, embeddings, 12,288 dimensions. Directions = meaning. King - man + woman = queen.
7. [ ] Core job: predict the next word. "Just autocomplete" — but doing it well requires learning everything.
8. [ ] Transformer: alternating attention + MLP blocks, 96 times.
9. [ ] Attention: query, key, value. 96 heads x 96 layers = 9,216 operations. 1/3 of parameters.
10. [ ] MLPs: store facts. AND gates + knowledge injection. 2/3 of parameters. Superposition = exponential capacity.
11. [ ] Diffusion: noise → images/video. CLIP connects words to pictures. Guidance steers the process.
12. [ ] A brain in a jar can't do anything. This afternoon, we give it hands. Model + workflow engine + tools = agent.
