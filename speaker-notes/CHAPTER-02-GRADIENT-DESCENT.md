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

- 13,002 adjustable parameters; random settings = garbage output
- "Learning" = finding good settings; need a metric for "how wrong"
- Ideal output for a 3: "3" neuron at 1.0, all others near 0
- Random weights: "3" at 0.2, "8" at 0.6 — clearly wrong, but need a number to quantify it
- That number = the **cost**

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

- Cost for one image = sum of squared differences between actual and desired output
- Example: "3" desired 1.0, actual 0.2 → (0.8)^2 = 0.64; "8" desired 0.0, actual 0.6 → (0.6)^2 = 0.36
- Squaring: makes all positive + amplifies big errors over small ones
- Average cost across all 60,000 training images = single performance score
- Cost function: takes 13,002 parameters in, spits one number out
- Think of it as a landscape — each parameter combo is a location, cost is the elevation; find the lowest valley

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

- Simplest case: one parameter (x-axis) vs. cost (y-axis) — a wavy curve
- Start at random point = random initial parameter value
- Blindfolded: can only feel the slope (the derivative) at current position
- Slope = direction of steepest ascent; go opposite = steepest descent
- Step downhill, recompute slope, step again, repeat until valley
- This is gradient descent in 1D — the core idea

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

- Cost landscape has multiple valleys — where you end up depends on where you start
- Global minimum = absolute best; local minimum = nearby valley, not necessarily the deepest
- Different random starts → different solutions
- In practice, local minima are fine — with 13,000+ dimensions, most local minima perform well
- No guarantee of global optimum, but good enough for real-world use

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

- Step size proportional to slope steepness — steep = big step, gentle = tiny step
- Naturally slows down near the minimum; prevents overshooting
- Fixed step size would bounce back and forth across the valley
- Slope encodes two things: **sign** = direction, **magnitude** = step size
- Self-regulating: no need to manually tune step sizes

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

- Two parameters → cost is a surface (like a topo map) instead of a curve
- Gradient = 2D vector pointing in direction of steepest ascent
- Negative gradient = direction of steepest descent — that's where you step
- Both parameters adjusted simultaneously each step
- Same concept as 1D but with a full directional arrow instead of just left/right

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

- Gradient = vector of 13,002 numbers, one per parameter
- Each component's **sign** = direction (increase or decrease), **size** = sensitivity
- Component of 3.2 vs 0.1 means cost is 32x more sensitive to the first parameter
- Gradient encodes "relative importance of each weight and bias"
- Small component = parameter barely matters; large component = high-leverage knob

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

- Gradient = ranked priority list of which adjustments matter most
- Not all 13,002 parameters adjusted equally — focus on high-leverage ones
- Some connections heavily influence output; others barely matter — gradient identifies which
- Scales to billions of parameters: every step targets the most impactful adjustments
- Without gradient: random knob-turning; with gradient: precision targeting

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

- ~96% accuracy after training — impressive but not perfect
- Hidden layer weights visualized: splotchy/noisy, not the clean edge detectors we hoped for
- Network found a strategy that works but is not human-interpretable
- Recurring AI theme: solution works, but we cannot explain exactly how
- Modern CNNs do learn clean edges; this simple 2-layer network is just the starting point

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

- 10 output neurons = always picks a digit; no 11th "not a digit" option
- Feed random noise → still outputs a confident digit classification
- Network "has no notion of the concept of a digit in general"
- Design limitation, not training failure — architecture forces a classification
- Same problem in modern AI: models can be confidently wrong (LLM hallucinations = same issue at scale)

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

- Full gradient requires processing all 60,000 images for one step — too expensive
- SGD: shuffle data, grab mini-batch of ~100, compute gradient, step, repeat
- Each step is approximate/noisy but fast; overall trend is still downhill
- Analogy: drunk man (fast, wobbly steps) beats careful man (slow, precise steps) to the bottom
- Every modern AI model trains this way — GPT, Claude, image models — all SGD with mini-batches

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

- Cost function = single number grading overall performance
- Gradient descent = compute slope, step downhill, repeat
- Gradient vector = 13,002 numbers saying which parameters to adjust and how much
- SGD = mini-batches of ~100 for fast approximate steps
- Same algorithm behind every AI model — architecture and scale change, learning loop stays the same
- Next: backpropagation — how to efficiently compute those 13,002 gradient values

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
