# Chapter 3 — Backpropagation: How the Network Assigns Blame

## Presentation Guide

**Duration:** 30-35 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build intuition for how backpropagation works — tracing errors backward through the network to figure out which weights and biases to adjust, and by how much — so the audience understands the complete learning loop from cost function to parameter update.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 3: What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **What backpropagation does** | 5 min |
| 0:05 | **One training example: what does it want?** | 5 min |
| 0:10 | **Three levers: bias, weights, previous activations** | 7 min |
| 0:17 | **Propagating backward through layers** | 5 min |
| 0:22 | **Averaging across examples** | 5 min |
| 0:27 | **Stochastic gradient descent and training data** | 5 min |
| 0:32 | **Recap** | 3 min |

**Total: ~35 minutes**

---

## What Backpropagation Does (5 min)

### SLIDE 01 — The gradient computation problem

**Headline:** Gradient descent tells you to "go downhill." Backpropagation tells you which way downhill actually is.

**Body:**
- In Chapter 2, we learned that the gradient is a list of 13,002 numbers — one per parameter
- Each number says how sensitive the cost is to that parameter
- But how do you actually compute those 13,002 numbers?
- **Backpropagation** is the algorithm that does this efficiently — it computes the gradient

**Visual concept:** A two-step diagram. Step 1 (labeled "Backpropagation"): compute the gradient arrow. Step 2 (labeled "Gradient descent"): step in that direction. Backpropagation feeds into gradient descent.

**Key insight card:**
Gradient descent is the strategy (go downhill). Backpropagation is the tactic (figure out which direction is downhill). You need both.

**Speaker notes:**

- Ch2 gave us the gradient (13,002 numbers) and gradient descent (step downhill)
- Missing piece: how do you actually compute those 13,002 sensitivities?
- Each one answers: "if I nudge this weight slightly, how much does cost change?"
- **Backpropagation** = the algorithm that computes all gradients efficiently
- Propagates information backward through the network; popularized 1986 (Rumelhart, Hinton, Williams)
- Still the backbone of all neural network training today

---

### SLIDE 02 — Focus on one training example

**Headline:** To understand backpropagation, start with a single image — a handwritten 2.

**Body:**
- Forget the full dataset for now — focus on just one training image
- Feed in a picture of a "2" and look at the output
- The output is messy: activations scattered across all 10 output neurons
- Ask: "What would this single example like the network to do differently?"

**Visual concept:** A handwritten "2" image entering the network on the left. On the right, a bar chart of the 10 output neurons with scattered activations. The "2" neuron is low. Arrows showing "this one should go UP" (green arrow on the 2 neuron) and "these should go DOWN" (red arrows on the others).

**Speaker notes:**

- Focus on one training image (a "2") — feed it through, look at output
- Output is scattered; "2" neuron not the strongest
- Key question: what does this example want changed?
- "2" neuron should go up; all other neurons should go down
- Strength of desire proportional to how far off each neuron is
- Starting point for backprop: begin at the output, work backward

---

## Three Levers: Bias, Weights, Previous Activations (7 min)

### SLIDE 03 — Three ways to change a neuron's activation

**Headline:** To increase the "2" neuron's output, you have three options.

**Body:**
1. **Increase the bias** — directly lower the threshold so the neuron fires more easily
2. **Increase the weights** — strengthen the connections from active neurons in the previous layer (proportional to activation — bigger adjustments for brighter neurons)
3. **Change the previous layer's activations** — ask the neurons feeding into it to change their values (proportional to weights — stronger connections get louder requests)

**Visual concept:** A single output neuron ("2") with multiple incoming connections from the previous layer. Three callout boxes, one per lever: (1) a bias slider being pushed up, (2) weight arrows getting thicker on connections from bright neurons, (3) activation values in the previous layer shifting up or down.

**Speaker notes:**

- Want the "2" neuron's activation to go up; formula: activation = squish(weighted sum + bias)
- **Lever 1 — Bias:** raise it → neuron fires more easily; blunt, input-independent
- **Lever 2 — Weights:** adjust proportional to previous layer's activations; bright neuron (0.9) = high leverage, dim neuron (0.01) = negligible
- **Lever 3 — Previous activations:** can't directly change them, but note what you *wish* they were; desire strength proportional to the weight on the connection
- Lever 3 = the "propagating backward" part — passing desires to the previous layer

---

### SLIDE 04 — Weights proportional to activations: neurons that fire together wire together

**Headline:** The biggest weight adjustments happen on connections from the brightest neurons.

**Body:**
- A neuron with activation 0.9 has 90x more leverage than one with activation 0.01
- The learning rule: adjust weights in proportion to the activation of the sending neuron
- This mirrors Hebb's rule from neuroscience: "neurons that fire together wire together"
- The strongest connections form between neurons that are frequently active at the same time

**Visual concept:** Two connections to the "2" output neuron. Connection A comes from a bright neuron (activation 0.9) — the weight adjustment arrow is large. Connection B comes from a dim neuron (activation 0.01) — the weight adjustment arrow is tiny. A caption: "Hebbian learning: fire together, wire together."

**Analogy card:**
In a team, you give more responsibility to the people who are already contributing the most. A team member sitting silent in the corner does not get promoted — the one who is actively delivering results gets a bigger role. That is how weight adjustments work.

**Speaker notes:**

- Weight adjustment proportional to sending neuron's activation
- Bright upstream neuron = big effect from weight change; dim = nearly irrelevant
- This IS Hebb's rule: "neurons that fire together wire together"
- Showing a 2 strengthens connections from neurons that activate on 2-like features
- Over many examples, strong connections form along paths that matter for each digit

---

### SLIDE 05 — Every output neuron has desires

**Headline:** It is not just the "2" neuron — every output neuron wants changes, and their desires get added together.

**Body:**
- The "2" neuron wants its activation to go up — requesting specific changes to previous layer weights and activations
- The "3" neuron (if it was incorrectly high) wants its activation to go down — requesting opposite changes
- Every output neuron broadcasts its desired changes to the previous layer
- **Add all those desires together** — the sum of all requests tells the previous layer what to do

**Visual concept:** Ten output neurons, each with arrows flowing backward to the previous layer. Different colored arrows represent different (sometimes conflicting) desires. A "sum" box where all the arrows merge into one combined signal per previous-layer neuron.

**Speaker notes:**

- All 10 output neurons have desires, not just the "2" — each broadcasts requests backward
- "2" wants up, "8" (if incorrectly high) wants down, "0" (already low) barely cares
- All 10 sets of requests summed together per previous-layer neuron
- Agreements reinforce; conflicts cancel — strongest combined signal wins
- Sum = the net desired change for each neuron in the previous layer

---

## Propagating Backward Through Layers (5 min)

### SLIDE 06 — The same process, layer by layer

**Headline:** Apply the same logic backward through every layer — that is the "back" in backpropagation.

**Body:**
- You now know what the last hidden layer's activations "should" be
- Use the same three levers to figure out how to get there: adjust biases, adjust weights, propagate desires to the layer before
- Repeat until you reach the input layer
- At the end, you have a desired adjustment for every weight and every bias in the entire network

**Visual concept:** The full network diagram with arrows flowing right to left through each layer. At each layer, the same three operations are shown: adjust biases, adjust weights, propagate backward. The leftmost layer (input) has no further propagation — the process terminates.

**Analogy card:**
Like a military after-action review that traces backward through the chain of command. The mission failed (high cost). Who made the final call? What intel did they have? Who gave them that intel? Was the original data collection flawed? Each link in the chain gets evaluated and adjusted.

**Speaker notes:**

- Recursive: output desires → last hidden layer desires → layer before that → ... → input
- Same three levers at each layer: adjust biases, adjust weights, propagate desires backward
- Stops at input layer (raw pixels are fixed)
- End result: desired adjustment for every weight and bias = the gradient (13,002 numbers)
- After-action review analogy: trace blame backward through chain of command, adjust each link

---

## Averaging Across Examples (5 min)

### SLIDE 07 — One example gives biased advice

**Headline:** A single training image creates adjustments biased toward that specific example. You need thousands.

**Body:**
- The picture of a 2 says: "Strengthen the '2' pathway, weaken the '8' pathway"
- A picture of an 8 would say the exact opposite: "Strengthen the '8' pathway, weaken the '2' pathway"
- If you only listen to one example, the network becomes a specialist for that one image
- **Average the desired adjustments across many examples** — the common signal survives, the noise cancels out

**Visual concept:** Three training images (a 2, an 8, a 5) each producing a set of adjustment arrows. Some arrows agree (pointing the same direction), others conflict. Below: the averaged result, where conflicting arrows cancel and agreeing arrows reinforce. Caption: "Individual adjustments are noisy. The average reveals the signal."

**Key insight card:**
Each training example is like one witness to an event. Any single witness might be biased or have a limited view. Interview a thousand witnesses and average their accounts — the truth emerges.

**Speaker notes:**

- One example's gradient is biased toward that specific image — potentially harmful to other digits
- Solution: average gradients across many training examples
- Conflicting signals cancel out; shared useful patterns survive
- Bridge: backprop computes gradient for one example → average many → gradient descent steps in that direction

---

## Stochastic Gradient Descent and Training Data (5 min)

### SLIDE 08 — Mini-batches: the practical compromise

**Headline:** Processing all 60,000 images for one step is too slow. Use random batches of ~100.

**Body:**
- Computing the gradient from every training image gives you the "true" gradient — but is astronomically expensive
- **Stochastic gradient descent:** Randomly sample a mini-batch of ~100 images, compute and average their gradients, take one step
- Each step is approximate — noisier than the true gradient — but much, much faster
- Over thousands of mini-batches, the network converges to a good solution

**Visual concept:** Two paths down a hillside. Path A (smooth, precise, labeled "Full dataset — one slow step"): a single long arrow pointing directly downhill. Path B (wobbly, fast, labeled "Mini-batches — many quick steps"): a zigzag path of short arrows that reaches the bottom faster despite the wobble.

**Analogy card:**
"A drunk man stumbling down a hill, taking quick, unsteady steps" vs. "a carefully calculating man determining the exact steepest descent and taking slow, deliberate steps." The drunk man gets to the bottom much faster.

**Speaker notes:**

- Full-batch gradient: process all 60,000 images per step — computationally impossible at scale
- SGD: shuffle, grab mini-batch (~100), backprop each, average gradients, step, repeat
- Each step is approximate (sample, not full dataset) but fast
- Like polling: random sample of 100 is noisy but reveals the trend
- Drunk man (SGD, fast wobbly steps) beats careful man (full-batch, slow precise steps) to the bottom
- Every AI model trained this way: mini-batch → backprop → gradient → step → repeat

---

### SLIDE 09 — You need lots of labeled training data

**Headline:** All of this only works if you have thousands of correctly labeled examples.

**Body:**
- The network learns from examples — no examples, no learning
- Each example must be **labeled** — someone has to say "this image is a 2, this one is a 7"
- MNIST: 60,000 training images of handwritten digits, each labeled by hand
- The quality and quantity of training data is often the bottleneck — not the algorithm
- More data = better gradients = better learning

**Visual concept:** A grid of handwritten digits with labels beneath each one (2, 7, 3, 9...). An arrow from the grid into the training loop. Caption: "60,000 labeled images — the fuel for learning."

**Speaker notes:**

- Cost function requires labeled data — can't measure "how wrong" without knowing "right"
- MNIST: 60,000 hand-labeled digit images (0-9)
- Language models: the "label" is the next word (self-labeling from raw text) — why they can train on the whole internet
- Supervised tasks (digit recognition) need manual labeling per example
- Quality matters: mislabeled data teaches wrong things; biased data → biased network
- Often the real bottleneck in AI: algorithms are published, data is the hard part

---

## Recap (3 min)

### SLIDE 10 — Chapter 3 recap: the complete learning loop

**Headline:** Backpropagation completes the loop — from measuring error to adjusting parameters.

**Body:**

| Step | What happens |
|------|-------------|
| 1. **Forward pass** | Feed an image through the network, get an output |
| 2. **Cost** | Measure how wrong the output is (sum of squared differences) |
| 3. **Backpropagation** | Trace errors backward — compute how each weight and bias contributed to the error |
| 4. **Gradient** | Collect all 13,002 sensitivities into a single vector |
| 5. **Step** | Nudge each parameter in the direction that reduces cost |
| 6. **Repeat** | Do it again with the next mini-batch, thousands of times |

**Visual concept:** A circular diagram showing the six steps as a continuous loop: Forward pass -> Cost -> Backpropagation -> Gradient -> Step -> next mini-batch -> Forward pass again. Caption: "This loop is the engine of all AI learning."

**Key insight card:**
This loop — forward pass, cost, backpropagation, gradient, step, repeat — is the engine behind every neural network ever trained. From a 13,002-parameter digit recognizer to a 175-billion-parameter language model. Same algorithm. Different scale.

**Speaker notes:**

- Complete loop: forward pass → cost → backprop → gradient → step → repeat with next mini-batch
- Same loop powers everything: digit recognizers, image generators, LLMs, self-driving cars
- AI is not magic — systematic: measure error, compute gradient, take a step, repeat millions of times
- Chs 2+3 together cover: cost function, backpropagation, gradient descent, SGD — the full learning process

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| What backpropagation does | 2 (slides 01-02) | 5 min |
| Three levers | 3 (slides 03-05) | 7 min |
| Propagating backward | 1 (slide 06) | 5 min |
| Averaging across examples | 1 (slide 07) | 5 min |
| SGD and training data | 2 (slides 08-09) | 5 min |
| Recap | 1 (slide 10) | 3 min |
| **Total** | **10 slides** | **~30 min** |

---

## Visual Design Notes

Match the existing workshop deck's design language:
- Dark backgrounds, PANW branding, Cyber Orange accents
- Card-based layouts for key concepts
- "Analogy cards" and "Key insight cards" in accent-colored rounded rectangles
- Use visual diagrams heavily — this audience is non-technical, so every concept should have a visual
- No raw equations anywhere. If math is needed, express it as a diagram or a visual metaphor
- Speaker notes are written in conversational/first-person facilitator tone, bullet-point style

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by Grant Sanderson. Specific content drawn from:
- Chapter 3: ["What is backpropagation really doing?"](https://www.youtube.com/watch?v=Ilg3gGewQ5U) (2017)
