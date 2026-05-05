# Chapter 2 — Gradient Descent: How Neural Networks Learn

## Presentation Guide

**Duration:** 30-35 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build intuition for how a neural network measures its own mistakes and systematically improves itself — the cost function, gradient descent, and stochastic gradient descent — so that backpropagation (Chapter 3) has a foundation.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — Chapter 2: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) by Grant Sanderson

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **The cost function — grading the network** | 5 min |
| 0:05 | **Gradient descent in 1D — rolling downhill** | 5 min |
| 0:10 | **Local minima and step sizes** | 5 min |
| 0:15 | **Scaling to many inputs — the gradient vector** | 5 min |
| 0:20 | **Relative importance of each parameter** | 5 min |
| 0:25 | **Results, limitations, and stochastic gradient descent** | 10 min |

**Total: ~35 minutes**

---

## The Cost Function — Grading the Network (5 min)

### SLIDE 01 — What does "learning" even mean?

**Headline:** Before a network can learn, it needs a way to measure how wrong it is.

**Body:**
- Show the network a "3" — ideally, the "3" output neuron fires high, everything else stays low
- With random parameters, the output is garbage — wrong neurons light up, right neurons stay dim
- We need a single number that captures "how bad is this?" — that number is the **cost**

**Visual concept:** Side-by-side comparison: the ideal output (the "3" neuron at 1.0, all others near 0) vs. the actual output with random weights (scattered activations, "3" neuron low, "8" neuron high). A big red number above the bad output: "Cost = 14.7"

**Speaker notes:**

- The network has 13,002 adjustable knobs (weights and biases)
  - With random settings, the output is nonsense
  - "Learning" means finding the right settings — but first you need a way to measure how far off you are
- Show the network an image of a 3
  - Ideal: "3" neuron at 1.0, everything else near 0.0
  - Reality with random weights: "3" neuron might be at 0.2, "8" at 0.6, "5" at 0.4
  - Clearly terrible — but we need to put a number on "how terrible"
- That number is the **cost** — and computing it is the first step in learning

---

### SLIDE 02 — The cost function: sum of squared differences

**Headline:** Square the gap between what you got and what you wanted. Add them up.

**Body:**
- For each of the 10 output neurons: subtract actual activation from desired activation, then square it
- Squaring does two things: makes all values positive, and amplifies big errors more than small ones
- Add all 10 squared differences together = the cost for this one training example
- Average the cost across all training examples (tens of thousands) = the network's overall grade

**Visual concept:** A table showing 10 output neurons, their desired values, actual values, the difference, and the squared difference. The sum at the bottom labeled "Cost for this image." Below that, a diagram showing thousands of images averaging into one number: "Average cost."

**Key insight card:**
The cost function is like an inverted exam score. Zero is perfect — the network got everything right. A high number means it is confused. Training is about driving this number toward zero.

**Speaker notes:**

- Walk through the math with a concrete example:
  - "3" neuron desired: 1.0, actual: 0.2 — difference: 0.8, squared: 0.64
  - "8" neuron desired: 0.0, actual: 0.6 — difference: 0.6, squared: 0.36
  - Add up all 10 squared differences — that is the cost for this one image
- Why square?
  - Makes negatives positive — we just care about size of the error, not direction
  - Amplifies big errors — a neuron that is way off gets penalized more than one that is slightly off
- One image gives one cost, but you need the big picture
  - Average the cost across all 60,000 training images
  - That average is a single number: the network's overall performance score
- The cost function takes in 13,002 parameters and spits out one number
  - Think of it as a landscape: each combination of parameters is a location, and the cost is the elevation
  - We need to find the lowest valley — that is where the best parameter settings live

---

## Gradient Descent in 1D — Rolling Downhill (5 min)

### SLIDE 03 — Start simple: one input, one output

**Headline:** Imagine the cost depends on just one parameter. You are blindfolded on a hilly line.

**Body:**
- Picture a simple curve — cost on the vertical axis, one parameter value on the horizontal
- You start at a random point (random parameter = high cost)
- You cannot see the whole curve — you can only feel the slope under your feet
- If the slope tilts right, step left. If it tilts left, step right. Always step downhill.

**Visual concept:** A wavy 1D curve (like a cross-section of hills). A dot sitting on a slope. An arrow pointing downhill from the dot. Dotted path showing the dot taking several steps toward the valley floor.

**Analogy card:**
You are blindfolded on a hillside. You feel the ground tilt under your feet. You step in whatever direction goes downhill. That is gradient descent — except you are doing it in a space with 13,000 dimensions instead of one.

**Speaker notes:**

- Start with the simplest possible version — one parameter, one cost value
  - Draw a wavy curve — horizontal axis is the parameter value, vertical axis is the cost
  - You drop a ball at a random spot — that is your initial random parameter
- You cannot see the landscape — you are blindfolded
  - All you can do is feel the slope — is the ground tilting left or right?
  - The slope is the **derivative** — it tells you the direction of steepest ascent
  - Go the other way — that is the direction of steepest descent
- Take a small step in the downhill direction
  - Recompute the slope at your new position
  - Take another step
  - Repeat until you settle into a valley
- This is gradient descent in one dimension — the core idea

---

### SLIDE 04 — Local minima: you land in the nearest valley

**Headline:** No guarantee you find the deepest valley — you find the nearest one.

**Body:**
- The cost landscape has many valleys (local minima)
- Gradient descent settles into whichever valley is closest to where you started
- That valley might not be the absolute best (the global minimum)
- In practice, this turns out to be fine — local minima are usually good enough

**Visual concept:** A wavy 1D curve with multiple valleys of different depths. Two dots starting at different positions, each rolling into a different valley. One valley is deeper (global minimum), but the dot that started farther away settled into a shallower valley (local minimum). Both labeled.

**Speaker notes:**

- The landscape is not a simple bowl — it has hills, ridges, and multiple valleys
  - Where you end up depends on where you start
  - Two different random starting points can lead to two different solutions
- The valley you land in might not be the deepest one
  - The deepest valley is the "global minimum" — the absolute best parameter settings
  - You might land in a "local minimum" — a valley that is not the absolute best but is still pretty low
- Why this is OK in practice:
  - With 13,000+ parameters, the landscape is so complex that there are many good valleys
  - Most local minima in high-dimensional spaces turn out to perform well
  - Grant's point: "depending on which random input you start at, you may wind up in different local minima, with no real guarantee that the local minimum you land in is going to be the smallest possible value"
  - In real-world applications, the local minima you find are usually good enough

---

### SLIDE 05 — Step sizes: proportional to slope

**Headline:** Steep slope = big step. Gentle slope = small step. This naturally avoids overshooting.

**Body:**
- Make each step proportional to how steep the slope is
- On a steep slope: big steps — you are far from the bottom, move quickly
- Near the valley floor: tiny steps — the slope flattens out, you slow down naturally
- This prevents you from rocketing past the minimum and bouncing back and forth

**Visual concept:** The same 1D curve, but now the arrows change size. On the steep part, the arrow is long. Near the bottom, the arrow shrinks. Labels: "Steep = big step" and "Flat = small step." A contrasting "bad" version shows oversized steps bouncing back and forth across the valley.

**Key insight card:**
The sign of the slope tells you which direction to go. The magnitude of the slope tells you how big a step to take. Both are encoded in one number.

**Speaker notes:**

- A natural and elegant feature of gradient descent:
  - Steep slope = you are far from the bottom, so take a big step
  - Gentle slope = you are close to the bottom, so take a tiny step
  - You naturally slow down as you approach the minimum
- Why this matters:
  - If your steps are always the same size, you might overshoot the valley and bounce back and forth forever
  - Proportional steps prevent that — "if the slope is smaller, each step should be smaller as well"
- The slope gives you two pieces of information in one number:
  - **Sign** (positive or negative): which direction to step
  - **Size** (large or small): how big a step to take
- This is why gradient descent works so well — it self-regulates

---

## Scaling to Many Inputs — The Gradient Vector (5 min)

### SLIDE 06 — Two inputs: the gradient gives direction

**Headline:** With two parameters, the gradient is an arrow pointing uphill. Go the other way.

**Body:**
- With two parameters, the cost function is a surface — like terrain on a map
- The **gradient** is a 2D arrow (a vector) pointing in the direction of steepest ascent
- The **negative gradient** points in the direction of steepest descent — that is where you step
- Each step adjusts both parameters simultaneously

**Visual concept:** A 3D surface plot (cost landscape) with contour lines viewed from above. A dot on the surface with two arrows: a red arrow pointing uphill (gradient) and a green arrow pointing downhill (negative gradient). A dotted path showing the dot winding downhill toward the minimum.

**Speaker notes:**

- Now scale up from one parameter to two
  - With one parameter, the cost is a curve (1D)
  - With two parameters, the cost is a surface — like a topographic map
  - The gradient is no longer just a slope — it is a 2D arrow (a vector)
- The gradient vector points in the direction of steepest ascent
  - Just like on a real mountain: stand at any point, look around, find the steepest uphill direction
  - The gradient points that way
  - Take the negative of it — now you have the direction of steepest descent
- Take a small step in the negative gradient direction
  - Both parameters get adjusted simultaneously
  - Recompute the gradient at your new position, step again, repeat
- The concept is the same as 1D, just richer
  - Instead of "step left or right," you now have a full direction in 2D space

---

### SLIDE 07 — 13,000 parameters: the full gradient

**Headline:** The gradient is a list of 13,002 numbers — one per parameter.

**Body:**
- Each number in the gradient says two things:
  - **Direction (sign):** Should this parameter go up or down?
  - **Magnitude (size):** How sensitive is the cost to this particular parameter?
- A gradient component of 3.2 vs. 0.1 means the cost is 32x more sensitive to the first parameter
- The gradient "encodes the relative importance of each weight and bias"

**Visual concept:** A tall column of labeled parameters (Weight 1, Weight 2, Bias 1, Weight 3...) with arrows of varying sizes pointing left (decrease) or right (increase). Large arrows = high sensitivity. Small arrows = low sensitivity. Color-coded: green for increase, red for decrease.

**Speaker notes:**

- Our digit network has 13,002 parameters — weights and biases
  - The gradient is a list (a vector) of 13,002 numbers — one per parameter
  - Each number tells you exactly what to do with that parameter
- Two pieces of information per component:
  - The **sign** tells you the direction — should this weight go up or down to reduce cost?
  - The **size** tells you the sensitivity — how much does the cost care about this parameter?
- Think of it as a ranked priority list
  - "Here are your 13,002 knobs. Here is exactly which direction to turn each one, and here is how much each one matters relative to the others."
  - Some parameters barely affect the cost — small gradient components
  - Others are critical — large gradient components
- Grant's key phrase: the gradient "encodes the relative importance of each weight and bias"
  - It tells you which changes will give you the most bang for your buck

---

## Relative Importance of Each Parameter (5 min)

### SLIDE 08 — Which changes matter most?

**Headline:** The gradient is a ranked priority list: which knobs to turn first for the biggest improvement.

**Body:**
- Not all parameters are equally important
- Some weights barely affect the output — tweaking them does almost nothing
- Others are critical leverage points — a small change creates a big improvement
- The gradient tells you exactly which is which, so you spend your effort where it counts

**Visual concept:** A dashboard-style display with parameter names and "impact meters" of varying lengths. A few parameters have large meters (high impact), most have small meters (low impact). Caption: "The gradient tells you where to focus."

**Analogy card:**
Like a mechanic diagnosing a car problem. There are a hundred things you could adjust, but the gradient tells you: "This one bolt is responsible for 40% of the vibration. Start there."

**Speaker notes:**

- This is one of the most powerful aspects of gradient descent
  - You do not adjust all 13,002 parameters equally
  - The gradient gives you a precise ranking of which adjustments matter most
- Think about it from the network's perspective:
  - Some connections carry signals that heavily influence the output
  - Other connections carry signals that barely matter
  - The gradient automatically identifies the high-leverage connections
- This is also why gradient descent scales
  - Even with billions of parameters, the gradient tells you exactly where to focus
  - Without it, you would be turning random knobs and hoping for the best
  - With it, every step is targeted at the most impactful adjustments
- Bridge: "OK, so gradient descent sounds great — but does it actually work? What happens when you train the digit network?"

---

## Results, Limitations, and Stochastic Gradient Descent (10 min)

### SLIDE 09 — Results: 96% accuracy, but messy internals

**Headline:** The trained network classifies 96% of digits correctly — but the hidden layers look nothing like what we hoped.

**Body:**
- After training on 60,000 images with gradient descent, the network reaches ~96% accuracy
- We *hoped* the hidden layers would learn clean edge detectors and pattern recognizers
- Reality: the hidden layer weights look random — splotchy, noisy, no clean edges
- The network found a solution, but not the "human-interpretable" solution we expected

**Visual concept:** A grid of 16 squares, each showing what a hidden-layer neuron "looks for" (its weights visualized as a 28x28 pixel pattern). Instead of clean edges and curves, they look like random noise with vague splotches. Label: "What we hoped for: clean edges. What we got: this."

**Key insight card:**
The network found parameter settings that work, but they do not decompose the problem the way a human would. It solved the exam — but used a strategy no one anticipated.

**Speaker notes:**

- Training results: about 96% accuracy on test images after gradient descent
  - That is genuinely impressive — from random weights to recognizing handwriting
  - But 96% is not 100% — it still makes mistakes
- The surprise: look at what the hidden layers actually learned
  - Remember the hope from Chapter 1 — that Layer 1 would learn edges, Layer 2 would learn loops and curves
  - When you visualize the weights, they look almost random — splotchy, noisy, no clean structure
  - "Some vague patterns" but nothing like the clean edge detectors we hoped for
- What does this mean?
  - The network found a different strategy — one that works but is not human-interpretable
  - It solved the problem, just not the way we expected
  - This is a recurring theme in AI: the solution works, but we often cannot explain exactly how
- This result motivates more sophisticated architectures
  - Modern convolutional neural networks (CNNs) do learn clean edge detectors
  - The simple 2-layer network we are discussing is a starting point, not the final answer

---

### SLIDE 10 — The network has no concept of "I don't know"

**Headline:** Feed it random noise — it will confidently classify it as a digit.

**Body:**
- The network always outputs probabilities for the 10 digits — it has to pick something
- Show it a picture of a dog, random static, or a blank image — it still outputs a confident digit classification
- It has learned patterns in handwritten digits, but it has no ability to say "this is not a digit"
- The network is fundamentally an answer machine — it cannot express uncertainty about whether the question even applies

**Visual concept:** Three input images side by side — a random noise image, a photo of a dog, and a scribble — each with a confident output bar chart showing a specific digit at high probability. Caption: "The network sees digits everywhere."

**Speaker notes:**

- A critical limitation worth understanding
  - The network has 10 output neurons — one per digit
  - It always assigns probabilities across those 10 options
  - There is no 11th output for "this is not a digit"
- Feed it random noise — it will say "that's a 5" with high confidence
  - It has no mechanism for saying "I don't know" or "this doesn't look like any digit"
  - Grant's point: the network "has no notion of the concept of a digit in general"
- Why this matters for real-world AI:
  - Same issue shows up in modern AI — models can be confidently wrong
  - A language model generating plausible-sounding nonsense is the same problem at a larger scale
  - Understanding this limitation is important when deploying AI in real systems
- This is a design limitation, not a training failure
  - The architecture forces a classification — it does not allow for "none of the above"
  - More sophisticated systems build in uncertainty estimation, but the basic architecture does not have it

---

### SLIDE 11 — Stochastic gradient descent: the practical shortcut

**Headline:** Instead of computing the gradient from all 60,000 images, use random batches of ~100.

**Body:**
- Computing the exact gradient requires processing every training image — astronomically expensive
- **Stochastic gradient descent (SGD):** Grab a random mini-batch of ~100 images, compute the gradient from just those, take a step
- Each step is noisier — not the "true" gradient, just an approximation from a sample
- But each step is fast, and the overall trend is still downhill
- Repeat with new random batches thousands of times

**Visual concept:** Two paths down a hillside. Path A (labeled "Full-batch gradient descent"): a smooth, precise path straight to the bottom — slow, careful. Path B (labeled "Stochastic gradient descent"): a wobbly, zigzag path that still reaches the bottom — fast, messy. A clock icon showing Path B is much faster.

**Analogy card:**
"A drunk man stumbling down a hill, taking quick, unsteady steps" vs. "a careful, calculating man determining the exact direction of steepest descent and taking slow, precise steps." The drunk man gets to the bottom faster.

**Speaker notes:**

- The practical problem with "pure" gradient descent:
  - To compute the true gradient, you need to process every single training image
  - For 60,000 images, that is one gradient computation = one step
  - Absurdly expensive for larger datasets — imagine billions of examples
- Stochastic gradient descent is the solution everyone uses:
  - Randomly shuffle the training data
  - Grab a mini-batch — about 100 examples
  - Compute the gradient from just those 100, take a step
  - Grab another 100, compute, step, repeat
- Each step is not the "true" gradient — it is an approximation
  - Sometimes it points slightly wrong — noisy, imperfect
  - But on average, it points in roughly the right direction
  - And each step is incredibly fast compared to processing the full dataset
- Grant's analogy: "a drunk man stumbling aimlessly down a hill but taking quick steps" vs. "a carefully calculating man taking slow, precise steps downhill"
  - The drunk man gets to the bottom faster despite the wobble
  - Each individual step is suboptimal, but the speed more than compensates
- This is how every modern AI model trains
  - GPT-3, Claude, every image model — all use stochastic gradient descent
  - Mini-batches, noisy steps, fast iterations, overall trend downhill

---

### SLIDE 12 — Chapter 2 recap

**Headline:** Four ideas that power all of AI learning.

**Body:**

| Concept | One-liner |
|---------|-----------|
| **Cost function** | Measures how wrong the network is — zero is perfect |
| **Gradient descent** | Step downhill in parameter space to reduce cost |
| **The gradient vector** | A ranked priority list: which knobs to turn, in which direction, and how much each one matters |
| **Stochastic gradient descent** | Use random mini-batches instead of the full dataset — noisy but fast |

**Visual concept:** Four icons in a row: a thermometer (cost), a ball rolling downhill (gradient descent), a ranked list (gradient vector), a zigzag path (SGD).

**Speaker notes:**

- Quick recap of the four core ideas:
  1. **Cost function** — a single number that grades the network's performance across all training examples
  2. **Gradient descent** — compute the slope, step downhill, repeat until you reach a valley
  3. **The gradient** — 13,002 numbers that tell you exactly which parameters to adjust and by how much
  4. **Stochastic gradient descent** — use mini-batches of ~100 examples for fast, approximate steps
- These four ideas are the engine behind every AI model ever trained
  - The architecture changes, the scale changes, but the learning algorithm is the same
  - Measure error, compute gradient, take a step, repeat
- Next up: backpropagation — the algorithm that actually computes the gradient efficiently
  - "How do you figure out, for each of those 13,002 parameters, exactly how it affects the cost?"

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| Cost function | 2 (slides 01-02) | 5 min |
| Gradient descent 1D | 3 (slides 03-05) | 5 min |
| Scaling to many inputs | 2 (slides 06-07) | 5 min |
| Relative importance | 1 (slide 08) | 5 min |
| Results & limitations | 3 (slides 09-11) | 10 min |
| Recap | 1 (slide 12) | 2 min |
| **Total** | **12 slides** | **~32 min** |

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
- Chapter 2: ["Gradient descent, how neural networks learn"](https://www.youtube.com/watch?v=IHZwWFHWa-w) (2017)
