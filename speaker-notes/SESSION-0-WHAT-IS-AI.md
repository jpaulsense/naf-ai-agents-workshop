# Session 0 — What Is AI and How Does It Work?

## Presentation Guide

**Duration:** 2 hours (Day 1, first block — before the LangGraph workshop)
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build genuine intuition for what AI is, how neural networks learn, and how the technology behind ChatGPT/Claude works — so the hands-on agent workshop has a foundation.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **Opening: Why this matters** | 5 min |
| 0:05 | **Block 1: What is a neural network?** | 25 min |
| 0:30 | **Block 2: How neural networks learn** | 25 min |
| 0:55 | **Break** | 10 min |
| 1:05 | **Block 3: From images to language — what is a GPT?** | 25 min |
| 1:30 | **Block 4: Attention — how AI understands context** | 20 min |
| 1:50 | **Block 5: How AI stores knowledge** | 5 min |
| 1:55 | **Closing: Bridge to the agent workshop** | 5 min |

**Total: 120 minutes**

---

## Opening: Why This Matters (5 min)

### SLIDE 01 — Title

**Title:** What Is AI and How Does It Work?

**Subtitle:** A visual, no-math tour of the technology behind ChatGPT, Claude, and modern AI.

**Speaker notes:**

- Two hours on how AI actually works — mechanics, not hype
- Goal: understand more than 99% of ChatGPT users by end of session
- Foundation for this afternoon's agent-building workshop
- Source material: 3Blue1Brown (Grant Sanderson) — 30M+ views on deep learning series
- No math background needed — ignore notation, listen for concepts

---

### SLIDE 02 — The one-sentence version

**Headline:** AI is a function that turns inputs into outputs — but instead of a human writing the rules, the machine learns them from examples.

**Visual concept:** Simple flow diagram: INPUT (image of a handwritten "3") → [BLACK BOX with "13,000 learned parameters"] → OUTPUT ("It's a 3")

**Speaker notes:**

- AI = mathematical function. Numbers in, numbers out.
- Not magic, not sentient — learned parameters define behavior
- Key difference from traditional code: machine learns its own rules from examples
- Random parameters = garbage. Right parameters = useful output.
- 13,002 parameters in our example network; 175B in GPT-3
- Our job: open the black box and understand what's inside

---

## Block 1: What Is a Neural Network? (25 min)

### SLIDE 03 — The problem: things humans do easily but can't explain

**Headline:** How do you recognize a handwritten "3"?

**Body:**
You do it instantly. But try to write step-by-step instructions for a computer:
- "Look for a curve at the top and bottom"
- But what counts as a curve? How curvy? What if it's sloppy?
- Every rule you write has a hundred exceptions

**Key insight card:**
Recognizing digits is incredibly easy for your brain to do, but almost impossible to *describe* how to do.

**Speaker notes:**

- Messy 3s on screen — audience recognizes instantly, but pixel values wildly different each time
- Traditional if/then rules fail: every rule has a hundred exceptions
- Neural networks solve tasks trivial for brains but impossible to code with rules
- Core ML insight: don't write rules — show examples, let machine learn rules
- Running example: 28x28 pixel handwritten digits — "hello world" of AI, scales to ChatGPT

---

### SLIDE 04 — What is a neuron?

**Headline:** A neuron is just a number.

**Body:**
- Not a brain cell — a container that holds a value between 0 and 1
- Think of it like a dimmer switch, not an on/off toggle
- 0 = completely off. 1 = fully activated. 0.5 = halfway
- That value is called the neuron's **activation**

**Visual concept:** A single circle with a number inside (e.g., 0.73), with a dimmer switch analogy beside it

**Speaker notes:**

- AI neuron = NOT a brain cell. Just a container holding a number between 0 and 1.
- That number = "activation." Dimmer switch, not on/off toggle.
- All of AI (ChatGPT, image gen, etc.) = networks of these simple number-holders
- Magic is in the connections, not any single neuron

---

### SLIDE 05 — Layers: breaking hard problems into easy steps

**Headline:** Stack neurons into layers. Each layer solves one piece of the problem.

**Visual concept:** Four vertical columns of circles (the network), left to right:
1. **Input layer** (784 neurons) — "Raw pixels"
2. **Hidden layer 1** (16 neurons) — "Edges & small shapes"
3. **Hidden layer 2** (16 neurons) — "Patterns & loops"
4. **Output layer** (10 neurons) — "Which digit?"

**Body:**
- **Input layer:** 784 neurons — one per pixel in a 28×28 image. Each holds the brightness of that pixel (0 = black, 1 = white).
- **Hidden layers:** The "thinking" happens here. These neurons learn to detect increasingly complex features — first edges, then loops, then combinations.
- **Output layer:** 10 neurons, one per digit (0-9). Whichever neuron activates most strongly is the network's answer.

**Analogy card:**
Think of it like a military chain of command. The front line (input) collects raw intel. Middle echelons (hidden layers) analyze patterns and synthesize. The top (output) makes the final call. Each level handles a different level of abstraction.

**Speaker notes:**

- Input: 784 neurons (28x28 pixels), each holds brightness 0-1
- Hidden: two layers of 16 neurons — the "thinking" layers
- Output: 10 neurons (digits 0-9); brightest = the answer
- Hierarchical decomposition: Layer 1 = edges/corners, Layer 2 = loops/patterns, Output = digit classification
- Examples: 9 = loop + right line; 8 = two loops; 4 = three lines
- Caveat: whether network actually learns this hierarchy is uncertain — but it's the hope
- Same pattern in speech: audio → sounds → syllables → words → meaning

---

### SLIDE 06 — How neurons connect: weights

**Headline:** Connections between neurons have *weights* — numbers that control influence.

**Visual concept:** Two neurons connected by a line, with a number on the line (e.g., +0.8 or -0.3). Show several connections with different weights, color-coded: blue = positive (excites), red = negative (inhibits). Also show: a 28×28 pixel grid with blue and red regions, illustrating a weight pattern that detects a specific edge.

**Body:**
- Every neuron in one layer connects to every neuron in the next
- Each connection has a **weight** — a number that says "how much should this input matter?"
- **Positive weight:** "If this neuron lights up, I should light up too"
- **Negative weight:** "If this neuron lights up, I should stay dark"
- **Large weight:** Strong influence. **Small weight:** Weak influence.

**Key insight card:**
Weights are where the "knowledge" lives. A trained network isn't a program — it's 13,000+ numbers that define how every neuron influences every other neuron.

**Speaker notes:**

- Weight = volume knob on each connection. Positive = excite, negative = inhibit, size = strength.
- Edge detection example: positive weights where edge should be, negative on surrounding pixels
- Can visualize what a neuron looks for by arranging its 784 weights into a 28x28 grid
- Every neuron connects to every neuron in next layer: 784 x 16 = 12,544 connections in first layer alone
- Total: 13,002 parameters (weights + biases) — those numbers ARE the network

---

### SLIDE 07 — Putting it together: the weighted sum + bias

**Headline:** Each neuron computes one simple thing: a weighted sum of its inputs, plus a bias, squished into range.

**Visual concept:** Diagram showing 3-4 input neurons flowing into one target neuron. Each arrow labeled with a weight. The target neuron shows the computation visually: multiply, add, bias, squish.

**Body:**
1. Multiply each incoming activation by its weight
2. Add them all up (the **weighted sum**)
3. Add a **bias** (a threshold — "how strong does the signal need to be before I care?")
4. Squish the result into the 0-to-1 range (the **activation function**)

**Speaker notes:**

- 4 steps: multiply inputs by weights → sum → add bias → squish into 0-1
- Weights = what pattern to look for. Bias = how strong the match must be to fire.
- Sigmoid: original activation function, smoothly squishes to 0-1. Grant: "squishification function"
- ReLU: modern replacement. Negative → 0, positive → unchanged. Avoids sigmoid's flat extremes.
- This one operation repeated thousands of times = all AI behavior

---

### SLIDE 08 — The network is just a function

**Headline:** A neural network is a function: 784 numbers in, 10 numbers out.

**Body:**
- Feed in pixel values → get back probabilities for each digit
- The function has 13,002 adjustable parameters (weights + biases)
- Different parameter values = different behavior
- The structure is designed by humans. The parameters are learned from data.

**Key insight card:**
There is nothing mystical here. A neural network is a mathematical function with a lot of knobs. The magic is in *how those knobs get set* — which is what we'll cover next.

**Speaker notes:**

- Demystification: neural network = mathematical function. 784 in → 10 out. Period.
- Compact notation: sigma(W*a + b). GPUs matter because matrix multiply is parallelizable.
- Random parameters = garbage. Right parameters = 98% accuracy.
- Humans design architecture; machine learns parameter values from data.
- Next: how do we find the right 13,002 settings? That's gradient descent.

---

## Block 2: How Neural Networks Learn (25 min)

### SLIDE 09 — The cost function: grading the network

**Headline:** To learn, the network first needs a way to measure how wrong it is.

**Visual concept:** Show a network outputting activations for digit "3": the "3" output neuron at 0.2 (low — bad!) and some wrong neurons like "8" at 0.6 (high — bad!). Red highlighting on the errors. Show the calculation: square the differences, add them up = cost for this example.

**Body:**
- Show the network an image of a "3" — it should output a high value for "3" and low for everything else
- The **cost function** measures the gap between what the network *did* output and what it *should have* output
- Low cost = the network got it right. High cost = the network is confused.
- Average the cost across thousands of training images = the network's overall "grade"

**Analogy card:**
The cost function is like an exam score — but inverted. Zero is perfect. The higher the number, the worse the performance. Training is about driving this score toward zero.

**Speaker notes:**

- Before improving, need to measure how bad: that's the cost function
- Show image of 3: ideal output = 1.0 for "3", 0.0 for everything else
- Cost = sum of squared differences between actual and ideal activations
- Average cost across all training images = single "grade" for the network
- Inverted score: 0 = perfect, high = confused
- Goal: find 13,002 parameter values that minimize this cost

---

### SLIDE 10 — Gradient descent: rolling downhill

**Headline:** Learning = finding the lowest point in a landscape of errors.

**Visual concept:** A 3D hilly landscape (like terrain). A ball sitting on a slope. An arrow pointing downhill. The valley floor labeled "minimum cost." Show the ball's path as a dotted line winding downhill through several steps.

**Body:**
- Imagine a landscape where elevation = cost (how wrong the network is)
- Each position in this landscape corresponds to a different set of parameter values
- The network starts at a random position (random parameters = high cost)
- **Gradient descent:** Figure out which direction is downhill → take a small step that way → repeat
- Each step adjusts the 13,000 parameters slightly to reduce the cost

**Key insight card:**
"Gradient" = the direction of steepest ascent. The *negative* gradient = the direction of steepest descent. That's where you step.

**Speaker notes:**

- Most important algorithm in all of AI — every model uses some version
- Intuition: cost function as landscape. Elevation = error. Goal = find lowest valley.
- Start at random position (random params). Blindfolded — can only feel local slope.
- Compute gradient (steepest uphill direction) → step opposite way → repeat billions of times
- No guarantee of finding the deepest valley (local minima) — but usually good enough in practice
- Learning rate = step size. Too big = overshoot. Too small = slow. Proportional to slope = natural deceleration near bottom.

---

### SLIDE 11 — What the gradient actually means

**Headline:** The gradient tells you which knobs matter most — and which direction to turn them.

**Visual concept:** A column of parameter names (Weight 1, Weight 2, Bias 1, Weight 3...) with arrows of different sizes pointing left or right. Large arrow = "this parameter matters a lot." Small arrow = "this one barely matters." Color-code: green arrows = increase this parameter, red arrows = decrease it.

**Body:**
- The gradient is a list of 13,002 numbers — one per parameter
- Each number says two things:
  - **Direction (sign):** Should this parameter go up or down?
  - **Magnitude (size):** How much does this parameter affect the cost?
- A gradient value of 3.2 vs. 0.1 means the cost is 32× more sensitive to that first parameter
- The gradient "encodes the relative importance of each weight and bias"

**Speaker notes:**

- Gradient = list of 13,002 numbers, one per parameter
- Sign tells direction: positive → decrease this param, negative → increase it
- Magnitude tells sensitivity: 3.2 vs 0.1 means first param is 32x more impactful
- Gradient = ranked priority list of which knobs matter most and which way to turn them
- Next question: how to compute it efficiently? That's backpropagation.

---

### SLIDE 12 — Backpropagation: tracing errors backward

**Headline:** Backpropagation = working backward through the network to figure out who's responsible for the error.

**Visual concept:** The same network diagram from Block 1, but with arrows flowing right-to-left (backward). Each arrow carries a "blame" signal. Show the flow: output error → hidden layer 2 → hidden layer 1 → weights.

**Body:**
- The output is wrong → how much did each neuron in the last hidden layer contribute to that error?
- Those neurons got their values from the layer before → how much did *those* neurons contribute?
- Keep going backward, layer by layer, assigning blame
- At each neuron, three things can be adjusted:
  1. The **bias** — directly shift the activation threshold
  2. The **weights** — change how much each input matters (bigger adjustments for brighter/more-active inputs)
  3. The **previous layer's activations** — propagate correction requests backward

**Analogy card:**
Like an after-action review. The mission failed (high cost). You trace backward through the chain of command: "Who made this decision? What intel did they have? Who gave them that intel?" Each link gets adjusted for next time.

**Speaker notes:**

- Backpropagation = propagate errors backward through the network to compute gradient
- Start at output: "2" neuron too low, "8" too high → what weights caused this?
- Key: adjustments proportional to activations. High activation = high leverage weight.
- Hebbian parallel: "neurons that fire together wire together"
- Also propagates desired activation changes backward — layer by layer assigning blame
- After-action review analogy: trace backward through chain of command
- Single example gives biased adjustments — need to average across many examples

---

### SLIDE 13 — Training at scale

**Headline:** One example teaches a little. Millions of examples teach a lot.

**Body:**
- A single training image creates one set of adjustments — biased toward that specific example
- **Stochastic gradient descent:** Process a random mini-batch of ~100 examples at a time
- Average their gradients → take one step → grab another batch → repeat
- Faster than processing everything at once, but noisier — like a "drunk man stumbling down a hill, but taking quick steps" vs. a careful person taking slow, precise steps

**Key insight card:**
Modern AI models train on billions of examples. GPT-3 was trained on roughly 300 billion tokens of text. Each token contributed a tiny nudge to 175 billion parameters.

**Speaker notes:**

- Stochastic gradient descent: random mini-batch (~100 examples), average gradients, step, repeat
- Grant's analogy: "drunk man stumbling down a hill taking quick steps" vs careful slow steps — noisy but fast
- Polling analogy: sample hundreds, not millions — noisy per sample but trends emerge
- Scale: digit network = 13,002 params, 60K images. GPT-3 = 175B params, 300B tokens. Same algorithm.
- Foundation complete: cost function → gradient → backprop → step → repeat
- Break, then applying this to language

---

## Break (10 min)

---

## Block 3: From Images to Language — What Is a GPT? (25 min)

### SLIDE 14 — The leap: from pixels to words

**Headline:** Same building blocks, different domain.

**Body:**
Everything we just learned applies:
- Neurons → still just numbers
- Layers → still stacked for hierarchical processing
- Weights → still learned from data via gradient descent
- The difference: instead of pixel brightness values, the input is **text**

**GPT stands for:**
- **G**enerative — it produces new text
- **P**re-trained — it learned from massive datasets before you ever used it
- **T**ransformer — the specific neural network architecture (the big innovation)

**Speaker notes:**

- Same fundamentals: neurons, layers, weights, gradient descent, cost function
- What changes: architecture (Transformer) and input format (text instead of pixels)
- G = Generative (produces new text). P = Pre-trained (learned before deployment). T = Transformer (the architecture).
- Transformer from 2017 Google paper "Attention Is All You Need" — most consequential ML paper ever

---

### SLIDE 15 — Tokens: breaking text into pieces

**Headline:** AI doesn't read words. It reads *tokens*.

**Visual concept:** The sentence "To date, the cleverest thinker of all time was..." broken into tokens with visible boundaries: "To| date|,| the| cle|ver|est| thinker| of| all| time| was|..."

**Body:**
- Text gets split into chunks called **tokens** — usually whole words or word fragments
- GPT-3's vocabulary: ~50,257 tokens
- Each token gets a unique ID number
- The model never sees raw text — only these numeric IDs

**Speaker notes:**

- Neural nets need numbers — tokenization converts text to numeric IDs
- Common words = one token. Uncommon/long words split into pieces ("cle" + "ver" + "est")
- GPT-3 vocabulary: ~50,257 tokens, each with unique ID
- Model never sees raw text — only numeric token IDs
- Context window = max tokens at once. GPT-3: 2,048. Modern models: 100K+.

---

### SLIDE 16 — Embeddings: giving words meaning

**Headline:** Each token becomes a long list of numbers that encodes its meaning.

**Visual concept:** The word "king" → an arrow pointing to a long column of numbers (12,288 values). Nearby in space: "queen," "monarch," "ruler." Far away: "bicycle," "sandwich."

**Body:**
- Each token ID gets looked up in a giant table called the **embedding matrix**
- The result: a vector (list) of 12,288 numbers for GPT-3
- These numbers encode the meaning of that word
- **Words with similar meanings end up close together** in this high-dimensional space

**Key insight card:**
The embedding isn't programmed by humans. The network *learns* which numbers to assign to each word during training. The structure that emerges is remarkable.

**Speaker notes:**

- Each token → embedding vector of 12,288 numbers (GPT-3). A point in high-dimensional space.
- Embeddings learned during training — not programmed by humans
- Similar meanings cluster together: king/queen close, cat/refrigerator far apart
- Structure emerges from billions of training examples — words in similar contexts converge
- Embedding matrix: 50,257 x 12,288 = ~618M params. Less than 0.5% of GPT-3's total.

---

### SLIDE 17 — Directions encode meaning

**Headline:** In embedding space, directions represent concepts.

**Visual concept:** A 2D simplified version showing:
- Arrow from "man" to "woman" labeled "gender direction"
- Same arrow from "king" to "queen"
- Same arrow from "uncle" to "aunt"
- Separate arrow from "Germany" to "Italy"
- Same arrow from "Hitler" to "Mussolini"

**Body:**
- The vector from "man" to "woman" captures a "gender" direction
- Add that direction to "king" → you land near "queen"
- Add it to "uncle" → you land near "aunt"
- "Italy" minus "Germany" plus "Hitler" → near "Mussolini"
- Directions for plurality: "cats" minus "cat" aligns with other plural nouns
- The network learned these relationships from text alone — no human labeled them

**Speaker notes:**

- "woman" - "man" = gender direction. "king" + gender direction = near "queen"
- Same direction: uncle → aunt, brother → sister. Consistent across all gendered pairs.
- "Italy" - "Germany" + "Hitler" = near "Mussolini" — nationality/role as separate directions
- Plurality direction: "cats" - "cat" dot-producted with "one","two","three","four" gives increasing values
- Nobody programmed this — model discovered abstract concepts from reading billions of sentences
- Language relationships encoded as geometry in 12,288-dimensional space

---

### SLIDE 18 — The core job: predict the next word

**Headline:** The entire model does one thing — predict what comes next.

**Visual concept:** A sequence of words: "The capital of France is ___" with a probability distribution showing "Paris" (68%), "a" (4%), "the" (3%), "located" (2%), etc. Then a Harry Potter example: "Harry Potter" + "least favorite" + "Professor" → high probability on "Snape"

**Body:**
- Given all the text so far, the model outputs a probability for every possible next token (~50,000 options)
- It samples one (with some randomness), appends it, and repeats
- This is how ChatGPT generates text: one token at a time, left to right
- **Temperature** controls randomness: low temperature = predictable/safe, high temperature = creative/wild

**Key insight card:**
"It's just autocomplete" — technically true. But to autocomplete well enough to write code, answer questions, and reason through problems, the model has to learn grammar, facts, logic, and common sense. The simple goal produces complex capability.

**Speaker notes:**

- Entire model trained for ONE task: predict the next token
- Output = probability distribution over ~50K vocabulary. Sample one, append, repeat.
- Example: "Harry Potter" + "least favorite" + "Professor" → high probability on "Snape"
- Temperature: low = conservative/deterministic, high = creative/random
- "Just autocomplete" — technically true, but to autocomplete well you need geography, law, logic, arithmetic, style...
- Simple goal produces complex capability. Predicting well requires learning everything about the world.

---

### SLIDE 19 — The Transformer architecture (high-level)

**Headline:** Text flows through the same two operations, repeated dozens of times.

**Visual concept:** A vertical pipeline showing tokens entering at the bottom, flowing upward through alternating blocks:
- ATTENTION BLOCK (blue)
- MLP BLOCK (green)
- ATTENTION BLOCK (blue)
- MLP BLOCK (green)
- ... (repeated ~96 times for GPT-3)
- UNEMBEDDING → SOFTMAX → PREDICTION at the top

**Body:**
- **Attention blocks:** Let each word look at every other word and update its meaning based on context
- **MLP blocks (feed-forward layers):** Process each word independently — add knowledge, refine understanding
- GPT-3 repeats this pair **96 times**
- The final vector passes through an **unembedding matrix** (the reverse of embedding) to produce probabilities

**Speaker notes:**

- Transformer = alternating blocks: Attention → MLP → Attention → MLP, repeated 96 times (GPT-3)
- Attention blocks: words talk to each other, update meanings based on context
- MLP blocks: process each word independently, inject factual knowledge
- Final vector → unembedding matrix → ~50K scores → softmax → probabilities → next token
- Embedding + unembedding = ~1.2B params. Remaining 174B in the 96 layers of attention + MLP.
- Next: attention — the mechanism that changed everything

---

## Block 4: Attention — How AI Understands Context (20 min)

### SLIDE 20 — The problem attention solves

**Headline:** The same word means different things in different contexts.

**Visual concept:** Three examples:
1. "American shrew **mole**" → animal
2. "One **mole** of carbon dioxide" → chemistry unit (6.022 × 10²³)
3. "Take a biopsy of the **mole**" → skin growth

Also show: "Eiffel **tower**" vs. "miniature **tower**" — same word, very different mental image

**Body:**
- Initial embeddings treat all instances of "mole" or "tower" identically — same 12,288 numbers every time
- But the meaning depends entirely on surrounding words
- **Attention** is the mechanism that lets each word look at its neighbors and update its meaning based on context

**Speaker notes:**

- Initial embeddings are context-free — "mole" gets same vector regardless of meaning
- Three meanings of "mole": animal, chemistry unit (6.022x10^23), skin growth — all start identical
- "Eiffel tower" vs "miniature tower" — same initial embedding, completely different meaning
- "Quill" in Harry Potter (pen) vs nature doc (hedgehog spine) — same vector, different meaning
- Attention = mechanism that lets surrounding words update each other's embeddings
- The key innovation in 2017 "Attention Is All You Need" paper

---

### SLIDE 21 — Attention in plain English

**Headline:** Each word asks: "Which other words should I pay attention to?"

**Visual concept:** The sentence "A **fluffy blue** creature roamed the **verdant** forest." Arrows from "creature" pointing back to "fluffy" and "blue" with strong connections. Arrows from "forest" pointing back to "verdant" with a strong connection.

**Body:**
How attention works, step by step:
1. Each word generates a **query** — "What am I looking for?" (nouns might ask: "Any adjectives describing me?")
2. Each word generates a **key** — "Here's what I have to offer" (adjectives might answer: "I'm a descriptor!")
3. Queries and keys get compared via dot product — "How relevant are you to me?"
4. Scores get normalized (softmax) so they sum to 1 — creating an **attention pattern**
5. High-scoring words contribute their **value** — the actual information to add
6. Each word updates its embedding with the gathered information

**Analogy card:**
Imagine a room full of people, each wearing a name tag describing what they know (key). You walk around with a question on your badge (query). You spend the most time talking to people whose name tags match your question — and you update your understanding based on those conversations.

**Speaker notes:**

- Example: "A fluffy blue creature roamed the verdant forest"
- Goal: let "fluffy" and "blue" update "creature" so it becomes "a fluffy blue creature"
- Query = what a word is looking for (nouns ask: "any adjectives near me?")
- Key = what a word has to offer (adjectives answer: "I'm a descriptor!")
- Dot product of query x key = relevance score. High = relevant, low = irrelevant.
- Softmax normalizes scores to sum to 1 — creates attention pattern
- Value = actual information to contribute. "Fluffy"'s value encodes the semantic content of fluffiness.
- High-scoring values get added to the word's embedding — updating it with context
- Networking event analogy: badge = key, lanyard = query, talk to highest-match people

---

### SLIDE 22 — Multi-head attention and masking

**Headline:** The network runs 96 attention passes in parallel — each looking for something different.

**Visual concept:** Multiple small attention-pattern grids showing different relationships: one tracking grammar (subject → verb), one tracking coreference (pronoun → noun), one tracking semantic theme. Below: a masking diagram showing a triangular pattern where future tokens are blacked out.

**Body:**
- Each "head" has its own query, key, and value matrices — learning different relationship types
- One head might learn grammar. Another might learn pronoun references. Another might track sentiment.
- GPT-3: 96 heads per layer × 96 layers = **9,216 attention operations total**
- ~58 billion parameters devoted to attention (one-third of GPT-3)
- **Masking:** During training, later words can't attend to future words (that would be cheating — seeing the answer before predicting it)

**Speaker notes:**

- One head = one set of Q/K/V matrices learning one relationship type
- 96 heads in parallel: grammar, pronoun resolution, semantic similarity, sentiment, etc.
- Example: one head resolves "it" in "animal didn't cross because it was tired" → "animal" not "street"
- All 96 heads' updates sum together — word enriched from 96 perspectives
- 96 heads x 96 layers = 9,216 total attention operations per token
- ~58B params devoted to attention (one-third of GPT-3)
- Masking: can't attend to future tokens (would be cheating). Scores set to -infinity → triangle pattern.
- Key success factor: attention is massively parallelizable on GPUs — enabled extreme scaling

---

### SLIDE 23 — Before and after attention

**Headline:** Attention transforms generic word vectors into context-rich representations.

**Visual concept:** Two columns:
- **Before (entering the network):** "king" = generic concept of any king ever
- **After 96 layers of attention + MLP:** "king" = a specific Scottish king who murdered his predecessor, written about in Shakespearean language in a play from ~1606, and the text is about to describe the psychological consequences
- Also show: "was" at the end of a mystery novel has been updated by attention to encode the entire mystery's worth of clues pointing to the murderer

**Body:**
- Input embeddings are context-free — just the dictionary definition
- After flowing through 96 layers, each vector encodes:
  - The word's meaning in *this specific context*
  - Relationships to other words in the passage
  - Accumulated knowledge from training data
- By the final layer, the last vector is rich enough to predict what comes next

**Speaker notes:**

- "King" enters as generic royalty concept. After 96 layers: "fictional Scottish king who murdered predecessor, play from 1606, psychological consequences incoming"
- All packed into same 12,288 numbers that started as just "king"
- Mystery novel example: "the murderer was..." — "was" by final layer encodes all clues, suspects, red herrings from entire context
- That enriched vector unembeds to probability distribution pointing at the correct suspect
- Attention transforms flat dictionary-definition vectors into deep contextually-rich representations

---

## Block 5: How AI Stores Knowledge (5 min)

### SLIDE 24 — Where do the facts live?

**Headline:** Facts are stored in the feed-forward (MLP) layers — the other half of each Transformer block.

**Visual concept:** The Transformer pipeline with MLP blocks highlighted. Expanded view of one MLP showing: Input vector → UP-PROJECTION (50,000 questions) → ReLU gate (yes/no) → DOWN-PROJECTION (add knowledge) → Updated vector. Concrete example: "Michael Jordan" → "Is this Michael + Jordan?" → YES → Add "basketball, Chicago Bulls, #23"

**Body:**
- Attention handles *relationships between words*
- MLP layers handle *knowledge and facts* — two-thirds of GPT-3's parameters
- Each MLP has ~50,000 "neurons" — each one a yes/no question about the input
- When a question matches, the associated knowledge gets added to the vector

**How it works (simplified):**
1. **Up-projection:** Multiply by a large matrix — each row asks a question via dot product ("Is this Michael + Jordan?"). Add a bias that sets a threshold (must match *both* names, not just one).
2. **ReLU gate:** Positive result = yes (active neuron). Negative = no (neuron stays at zero). This acts like an AND gate — it only fires when multiple conditions are met.
3. **Down-projection:** Each active neuron's column in a second matrix encodes the knowledge to add (e.g., the "basketball" direction). Active neurons add their knowledge; inactive neurons contribute nothing.
4. **Add back:** The result gets added to the original embedding — enriching it with facts.

**Speaker notes:**

- Attention = relationships between words. MLPs = knowledge/facts. MLPs hold 2/3 of GPT-3's params.
- Example: "Michael Jordan" flows in → MLP adds basketball, Chicago Bulls, #23
- Up-projection: ~50K rows each ask a yes/no question via dot product. Bias acts as AND gate.
- ReLU gate: positive = fire (yes), negative = zero (no). "Michael Jordan" fires, "just Michael" doesn't.
- Down-projection: each active neuron's column = knowledge to inject. Inactive neurons contribute nothing.
- Single neuron column can encode multiple facts simultaneously
- Scale: 1.2B params per MLP block x 96 layers = ~116B params (two-thirds of GPT-3)
- Superposition: features share neurons like overlapping radio frequencies
- Johnson-Lindenstrauss lemma: 12,288 dimensions can pack 40B+ nearly-independent directions
- Bigger models = exponentially more room for knowledge

---

## Closing: Bridge to the Agent Workshop (5 min)

### SLIDE 25 — Recap: the journey from neuron to GPT

**Headline:** What we just covered — nine concepts, from building block to breakthrough.

**Visual concept:** A horizontal timeline or staircase, left to right:

| Step | Concept | One-liner |
|------|---------|-----------|
| 1 | Neuron | A number between 0 and 1 — a dimmer switch |
| 2 | Layer | Break a hard problem into easy steps — each layer handles one level of abstraction |
| 3 | Weights & bias | Volume knobs that control influence + a minimum threshold |
| 4 | Cost function | A grade for how wrong the network is — zero is perfect |
| 5 | Gradient descent | Roll downhill to reduce error — billions of tiny steps |
| 6 | Backpropagation | Trace errors backward to assign blame — like an after-action review |
| 7 | Tokens & embeddings | Turn words into meaningful numbers — similar words land nearby |
| 8 | Attention | Let words look at each other for context — query, key, value |
| 9 | MLP layers | Inject learned facts and knowledge — two-thirds of the model |

**Speaker notes:**

- Recap the 9-step staircase on slide — point to each one briefly
- Hour 1: neuron → layers → weights/bias → cost function → gradient descent → backprop
- Hour 2: tokens/embeddings → attention (96 heads x 96 layers = 9,216 passes) → MLPs (2/3 of model)
- These 9 concepts = foundation of ALL modern AI: ChatGPT, Claude, image gen, self-driving cars

---

### SLIDE 26 — From brain to agent

**Headline:** A brain in a jar can't do anything. This afternoon, we give it hands.

**Body:**
Everything we just learned is about the **model** — the AI "brain." But a brain alone can only answer questions and generate text.

To make AI *useful*, you need:
- A **workflow engine** (LangGraph) — gives the brain a plan to follow
- **Tools** (Python functions) — gives the brain hands to act with
- A **decision loop** — lets the brain reason, act, observe, and adapt

That's what turns an AI model into an AI **agent** — and that's what we're building this afternoon.

**Speaker notes:**

- Everything so far = the brain. Brain alone can't configure firewalls or query databases.
- This afternoon: give the brain a body. Three components:
- Workflow engine (LangGraph) = defines steps, order, data flow
- Tools (Python functions) = specific actions: configure, query, create, check
- Decision loop = reason → act → observe → adapt. That's what makes it an "agent."
- When I say "LLM decides" in workshop: text tokenized → embedded → 96 layers of attention + MLP → predicts next tool
- You now understand the brain. Next: give it a body.

---

### SLIDE 27 — What you now know

**Headline:** You understand more about AI than 99% of people who use it daily.

**Body:**
- AI is not magic — it's a mathematical function with learned parameters
- Neural networks learn by measuring errors and rolling downhill (gradient descent)
- Modern language AI (GPT, Claude) uses the Transformer architecture
- Attention lets words understand context by looking at each other
- Facts and knowledge are stored as patterns in the MLP layers
- "It's just predicting the next word" — but doing that well requires learning everything about the world

**Bottom text:**
15-minute break, then we start building.

**Speaker notes:**

- Wrap: you now know how AI works — not just what, but how
- Dinner party answer: "function that predicts next word via dot products across 96 attention heads, trained via SGD on 300B tokens" [pause for laugh]
- This foundation makes the afternoon workshop click — you'll know what's under the hood
- 15-minute break, then we start building

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| Opening | 2 (slides 01-02) | 5 min |
| Block 1: Neural networks | 6 (slides 03-08) | 25 min |
| Block 2: Learning | 5 (slides 09-13) | 25 min |
| Block 3: GPT & language | 6 (slides 14-19) | 25 min |
| Block 4: Attention | 4 (slides 20-23) | 20 min |
| Block 5: Knowledge storage | 1 (slide 24) | 5 min |
| Closing | 3 (slides 25-27) | 5 min |
| **Total** | **27 slides** | **~110 min + 10 min break** |

---

## Visual Design Notes

Match the existing workshop deck's design language:
- Dark backgrounds, PANW branding, Cyber Orange accents
- Card-based layouts for key concepts
- "Analogy cards" and "Key insight cards" in accent-colored rounded rectangles
- Use visual diagrams heavily — this audience is non-technical, so every concept should have a visual
- No raw equations anywhere. If math is needed, express it as a diagram or a visual metaphor
- Speaker notes are written in conversational/first-person facilitator tone

---

## How This Fits the Two-Day Schedule

### Day 1 (4 hours)
| Block | Session | Duration |
|-------|---------|----------|
| 1 | **Session 0: What Is AI?** (this document) | 2 hr |
| — | Break | 15 min |
| 2 | Workshop Sessions 1-2: Foundations & State (NB 101-104) | 1 hr 45 min |

### Day 2 (4 hours)
| Block | Session | Duration |
|-------|---------|----------|
| 3 | Workshop Session 3: Conditional Routing (NB 106) | 45 min |
| — | Break | 10 min |
| 4 | Workshop Session 4: AI Integration & ReAct (NB 108, 110) | 1 hr |
| — | Break | 10 min |
| 5 | Beyond Network Security + Closing | 30 min |
| 6 | Open lab / Q&A | ~45 min |

---

## Facilitator Preparation Notes

### Handling common audience questions

**"Is AI conscious / alive / sentient?"**
> "No. What we just showed is that it's a mathematical function — matrix multiplications, dot products, and activation functions. It has no experiences, no awareness, no desires. It's extraordinarily good at predicting text, and that ability creates an *illusion* of understanding. But under the hood, it's weighted sums and gradient descent. Very impressive weighted sums and gradient descent — but that's what it is."

**"Is AI dangerous?"**
> Acknowledge genuinely, don't dismiss. "There are real concerns — about misuse, about bias in training data, about economic disruption. The best way to have an informed opinion about those risks is to understand how the technology actually works — which is what we just spent two hours doing. The policy questions are real, but they're outside our scope today."

**"Will AI take my job?"**
> "AI is a tool. Like every powerful tool in history, it changes what work looks like. The people who understand the tool are the ones who get to decide how it's used. That's part of why we're here."

**"How is Claude different from ChatGPT?"**
> "Same fundamental architecture — Transformers, attention, MLPs, trained via gradient descent. Different training data, different fine-tuning approaches, different companies. It's like asking how a Ford F-150 is different from a Chevy Silverado — same basic engineering, different execution."

**"How much does it cost to train these models?"**
> "GPT-3 cost an estimated $4-12 million to train. GPT-4 is rumored at $100+ million. The cost is mostly GPU compute time — running the gradient descent on billions of examples across thousands of specialized chips for weeks or months."

### Key numbers to have handy

| Stat | Value |
|------|-------|
| Digit network parameters | 13,002 |
| Input image size | 28×28 = 784 pixels |
| GPT-3 parameters | 175 billion |
| GPT-3 training tokens | ~300 billion |
| GPT-3 embedding dimension | 12,288 |
| GPT-3 vocabulary size | 50,257 tokens |
| GPT-3 layers | 96 |
| GPT-3 attention heads per layer | 96 |
| GPT-3 attention parameters | ~58 billion (1/3 of model) |
| GPT-3 MLP parameters | ~116 billion (2/3 of model) |
| Embedding + unembedding matrix | ~1.2 billion |
| MLP neurons per block | ~50,000 |
| GPT-3 context window | 2,048 tokens |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 1: "But what is a Neural Network?" (2017)
- Chapter 2: "Gradient descent, how neural networks learn" (2017)
- Chapter 3: "What is backpropagation really doing?" (2017)
- Chapter 5: "But what is a GPT? Visual intro to Transformers" (2024)
- Chapter 6: "Attention in transformers, step-by-step" (2024)
- Chapter 7: "How might LLMs store facts" (2024)
