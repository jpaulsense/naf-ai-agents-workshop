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

> Welcome everyone. Before we build anything with AI this afternoon, we're going to spend two hours understanding what it actually is. Not the hype you see in headlines, not the fear, not the sci-fi — the actual mechanics of how this technology works.
>
> By the end of this session, you're going to understand more about how AI works than 99% of the people who use ChatGPT every day. And that understanding is going to make everything we do this afternoon — when we start building AI agents — click in a way it wouldn't otherwise.
>
> A lot of the concepts and examples I'll be using today come from a YouTube channel called 3Blue1Brown, run by a guy named Grant Sanderson. He's one of the best explainers of math and AI on the internet — his videos on this topic have been watched over 30 million times. I've taken his key concepts and examples and built them into this presentation, adapted for this audience. If you want to go deeper on any of this after today, his channel is an incredible resource.
>
> Fair warning: you will NOT need any math background for this. If at any point you see a formula on a slide and your eyes glaze over — ignore it and listen to the explanation. The concepts are what matter, not the notation.

---

### SLIDE 02 — The one-sentence version

**Headline:** AI is a function that turns inputs into outputs — but instead of a human writing the rules, the machine learns them from examples.

**Visual concept:** Simple flow diagram: INPUT (image of a handwritten "3") → [BLACK BOX with "13,000 learned parameters"] → OUTPUT ("It's a 3")

**Speaker notes:**

> Everything we're about to cover for the next two hours comes back to this one slide. AI is not magic. It is not sentient. It is not "thinking" the way you and I think. At its core, AI is a mathematical function — a machine that takes in numbers, does a bunch of math, and spits out numbers.
>
> The thing that makes it special — the thing that makes it different from a regular computer program — is that a human didn't write the rules. Instead, the machine figured out its own rules by looking at millions of examples.
>
> That black box in the middle? It contains thousands — or in the case of ChatGPT, *billions* — of adjustable numbers called parameters. With random parameters, the function is useless garbage. But with the *right* parameters, it can recognize your face, understand your voice, write code, or have a conversation.
>
> Our job for the next two hours is to open that black box. We're going to understand what's inside, and how it gets set up. Let's start with the building blocks.

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

- Show messy handwritten 3s on screen — audience recognizes them instantly
  - Same concept, wildly different pixel values each time
  - The light-sensitive cells firing in your eye are completely different for each one, but your brain resolves them as the same idea
- Now try to *program* that with traditional if/then rules
  - "Two curves stacked, open to the left" — but what counts as a curve? How curvy? What about sharp-angled 3s?
  - Every rule has a hundred exceptions, and the exceptions have exceptions
- This is the problem **neural networks** were invented to solve
  - Tasks that are trivial for human brains but impossible to code with rules: images, speech, handwriting
- The core insight of **machine learning**: don't *tell* the computer the rules — *show* it thousands of examples and let it figure out the rules on its own
- Running example for this section: handwritten digit recognition (28×28 pixel images)
  - The "hello world" of AI — simple enough to understand, contains every concept that scales up to ChatGPT

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

- Forget biology — an AI neuron is NOT a brain cell
  - It's just a container that holds a number between 0 and 1
- The term for that number: **activation**
  - Dimmer switch analogy — not on/off, it's a range
  - 0 = off, 1 = fully lit, anything in between
- That's it — a neuron is a box with a number in it
  - Everything AI does — ChatGPT, image generation, recommendations — comes from networks of these simple number-holders interacting
  - The magic isn't in any single neuron, it's in how they're connected

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

- **Input layer**: 784 neurons — one per pixel in 28×28 image
  - Each holds pixel brightness (0 = black, 1 = white)
  - Just raw data, nothing fancy
- **Hidden layers**: two layers of 16 neurons each
  - Why 16? Somewhat arbitrary — the specific number matters less than the concept
  - These are the "thinking" layers
- **Output layer**: 10 neurons, one per digit (0–9)
  - Brightest neuron = the network's answer
  - "3" neuron at 0.95, everything else below 0.1 → network says "that's a 3"
- The hope for what hidden layers learn — **hierarchical decomposition**:
  - Layer 1 detects simple features: edges, small line segments, corners
  - Layer 2 combines edges into patterns: loops, vertical lines, crossbars
  - Output maps patterns → digits
- Specific digit examples (good for audience):
  - A 9 = loop on top + line on the right
  - An 8 = loop on top + another loop on the bottom
  - A 4 = three specific lines
  - Key caveat: "whether or not the network *actually* learns it this way is another question" — but this is the hope
- Same hierarchical pattern applies beyond images:
  - Speech: raw audio → distinct sounds → syllables → words → phrases → abstract thoughts
  - Military analogy: front line collects raw intel, middle echelons analyze, top makes the call

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

- A **weight** = a number on each connection between neurons
  - Volume knob analogy — controls how much influence one neuron has on the next
  - Positive weight: "if you light up, I should too"
  - Negative weight: "if you light up, I should stay quiet"
  - Size of weight = strength of influence
- Edge detection example — visualize weights as a pixel grid:
  - Positive weights (green/blue) on pixels where the edge should be
  - Negative weights (red) on surrounding pixels
  - Why negative? You want to detect a *pattern*, not just brightness — "the sum is largest when the middle pixels are bright but surrounding pixels are darker"
  - You can literally *see* what a neuron is looking for by arranging its weights into a 28×28 grid
- Scale: every neuron connects to *every* neuron in the next layer
  - 784 × 16 = 12,544 connections in just the first layer
  - Total across the whole network: **13,002 adjustable parameters** (weights + biases)
  - Those 13,002 numbers *are* the network — they define everything it knows

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

- Every neuron does the same 4-step computation:
  1. **Multiply** each incoming activation by its connection weight
  2. **Add** all those products together → the **weighted sum**
  3. **Add a bias** — a threshold that controls how strong the signal needs to be before the neuron cares
  4. **Squish** the result into the 0–1 range using an **activation function**
- **Bias** explained:
  - Weights tell you *what pattern* the neuron looks for
  - Bias tells you *how strong the match needs to be* before the neuron fires
  - Grant's phrasing: "how high the weighted sum needs to be before the neuron starts getting meaningfully active"
- **Activation functions** — two to name:
  - **Sigmoid function** (original): smoothly squishes any number into 0–1 range
    - Very negative → near 0, very positive → near 1
    - Grant calls it the "sigmoid squishification function"
  - **ReLU** (Rectified Linear Unit) — what modern networks use
    - Dead simple: negative → 0, positive → pass through unchanged
    - Why the switch? Sigmoid gets flat at extremes, so training gets stuck. ReLU never flattens.
- Bottom line: this one simple operation — multiply, sum, bias, squish — repeated thousands of times across layers, produces all the complex behavior in AI

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

- Key demystification moment — make this land:
  - A neural network is a **mathematical function**. Full stop.
  - 784 numbers in (pixels) → math in the middle → 10 numbers out (digit probabilities)
  - Can be written compactly: **σ(W·a + b)** — weight matrix × activations + biases, squished
  - This is why GPUs matter: matrix multiplication is massively parallelizable, and "many libraries optimize the heck out of matrix multiplication" (Grant's words)
- 13,002 **parameters** (weights + biases) = everything the network knows
  - Random parameters → complete garbage output
  - *Right* parameters → 98% accuracy on handwriting
- Humans design the **architecture** (layers, connections, activation functions)
  - The machine learns the **parameter values** from data
- Reassuring complexity: "if it were any simpler, what hope would we have that it could take on the challenge?" (Grant)
- Bridge to next section: "learning" = finding the right settings for all 13,002 knobs → that's **gradient descent**, coming up next

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

> OK, so we have 13,002 knobs and dials, and we need to find the right settings. But before we can improve the network, we need a way to measure how bad it currently is. That's what the cost function does.
>
> Here's how it works. You show the network an image of a 3. With random parameters, the output is garbage — maybe the "3" neuron has an activation of 0.2, but the "8" neuron is at 0.6, and the "5" neuron is at 0.4. That's terrible. The ideal output would be 1.0 for "3" and 0.0 for everything else.
>
> To measure how wrong this is, you take the difference between each output neuron's actual activation and what it *should* be, square those differences (so negatives become positive and big errors get amplified), and add them all up. That gives you the cost for this one training example.
>
> Now do that for every image in your training set — tens of thousands of labeled handwritten digits — and average all those individual costs together. That average is a single number that tells you how well or poorly the network is performing overall. The lower the cost, the better the network is doing.
>
> Think of it like an inverted exam score. Zero means perfect — the network got everything right. A high number means it's confused and making lots of mistakes. Our goal is to find values for those 13,002 parameters that make this cost as small as possible. And that brings us to the most important algorithm in all of AI.

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

> This is gradient descent, and it is the single most important algorithm in modern AI. Every AI model you've ever heard of — ChatGPT, Claude, Midjourney, self-driving cars, voice assistants — all learned through some version of this algorithm.
>
> Here's the intuition. Imagine the cost function as a physical landscape — hills, valleys, slopes, ridges. The elevation at any point represents the cost — how wrong the network is — for a particular set of parameter values. High elevation = high cost = bad performance. Low elevation = low cost = good performance.
>
> When you initialize a network with random parameters, you start at a random location in this landscape — probably somewhere up on a hill. The goal is to get to the lowest valley. But you can't see the landscape — you're blindfolded. All you can do is feel the ground under your feet and figure out which direction slopes downward.
>
> That's gradient descent. You compute the gradient — which tells you the direction of steepest *uphill* — and then you go the opposite direction. You take a small step downhill. Then you compute the gradient again from your new position. Take another small step. Repeat. Thousands, millions, billions of times. Eventually, you settle into a valley — a set of parameter values where the cost is low and the network performs well.
>
> One important subtlety: there's no guarantee you find the *deepest* valley. You might end up in a local minimum — a valley that's not the absolute best, but is still pretty good. In practice, this turns out to be fine for most real-world applications. The valleys you find are usually good enough.
>
> The step size matters too. If your steps are too big, you might overshoot the valley and bounce back and forth. If your steps are too small, training takes forever. The step size is controlled by something called the **learning rate**. And there's a nice built-in feature: if you make your step size proportional to the steepness of the slope, you naturally take smaller steps as you approach the bottom of a valley, reducing the risk of overshooting.

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

> Let me make the gradient more concrete, because it's one of those terms that sounds intimidating but is actually simple.
>
> Remember, our network has 13,002 parameters — weights and biases. The gradient is just a list of 13,002 numbers, one for each parameter. Each number in the gradient tells you two things:
>
> First, the **sign** — positive or negative — tells you which direction to nudge that parameter. Positive means "this parameter should go down to reduce cost." Negative means "this parameter should go up."
>
> Second, the **size** of the number tells you how sensitive the cost is to that parameter. If one gradient component is 3.2 and another is 0.1, that means the cost is 32 times more sensitive to the first parameter. A small change to the first weight makes a big difference; a small change to the second barely matters.
>
> So the gradient isn't just saying "go downhill." It's a ranked priority list: "Here are the 13,002 knobs. Here's exactly which direction to turn each one, and here's how important each one is relative to the others." It tells you which changes will give you the most bang for your buck.
>
> Now, the question is: how do you actually compute this gradient? How do you figure out, for each of those 13,002 parameters, how it affects the final cost? That's where backpropagation comes in.

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

> Backpropagation is the algorithm that computes the gradient efficiently. The name literally describes what it does: it propagates errors *backward* through the network.
>
> Let's walk through it. Say you show the network an image of a 2, and the output is wrong — the "2" neuron is barely active, and the "8" neuron is lit up. Clearly, some things need to change.
>
> You start at the output and work backward. The "2" neuron should be higher and the "8" neuron should be lower. OK — so what feeds into those output neurons? Neurons in the last hidden layer, through weighted connections. For each connection, you can figure out: "If I tweaked this weight a little bit, how much would it help?"
>
> Here's the key insight Grant makes in his videos: the adjustments should be **proportional to the activations**. If a neuron in the previous layer had a high activation — 0.9, say — then the weight on that connection has a lot of leverage. A small change to that weight creates a big change in the output. But if the previous neuron had an activation of 0.01, the weight barely matters — tweaking it does almost nothing. So you make bigger adjustments to weights connected to highly active neurons.
>
> This connects to something from neuroscience called Hebbian theory — "neurons that fire together wire together." The strongest learning happens along the connections between neurons that are both active. It's a nice parallel between artificial and biological neural networks.
>
> But you don't just adjust weights — you also figure out what the *previous layer's activations* should have been. "The '2' output neuron would be better if this hidden neuron were brighter and that one were dimmer." Those desired changes get propagated backward to the previous layer, where the same process repeats. Layer by layer, backward through the network, assigning blame and computing corrections.
>
> Think of it like a military after-action review. The mission failed — that's your cost. You trace backward through the chain of command: "Who made the final call? What intel were they working from? Where did that intel come from? Was the original data collection flawed, or was the analysis wrong?" Each link in the chain gets evaluated, and adjustments get made for next time.
>
> One more subtlety: a single training image — like one picture of a "2" — creates one set of desired adjustments. But those adjustments are biased toward that specific example. The picture of a 2 might say "make the '8' neuron quieter!" but a picture of an 8 would say the opposite. You need to average the desired adjustments across many examples to find a direction that's good for the overall dataset, not just one image. Which brings us to training at scale.

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

> So here's the practical reality. You don't process one image at a time — that would be too noisy, too biased toward individual examples. But you also don't process your entire training set at once — that would be astronomically expensive to compute.
>
> The practical solution is called **stochastic gradient descent**. You grab a random mini-batch — say, 100 images — compute the gradient for each one, average those gradients together, and take one step in that average direction. Then you grab another random 100, compute, average, step. Repeat millions of times.
>
> Grant Sanderson uses a great analogy for this: it's like "a drunk man stumbling aimlessly down a hill but taking quick steps, rather than a carefully calculating man who takes slow, deliberate steps downhill." Each individual step is a little random, a little noisy. But the overall trend is downhill, and you get there much faster because each step is so cheap to compute.
>
> It's like political polling. You don't survey every person in the country — that's too expensive. You poll random samples of a few hundred people. Each poll is a little noisy, but the trends emerge, and you can run polls much more frequently.
>
> Now let's talk about scale. The handwritten digit network we've been discussing has 13,002 parameters and trains on about 60,000 images. That's cute. GPT-3 — the model that powers ChatGPT — has 175 *billion* parameters. It was trained on approximately 300 *billion* tokens of text — books, websites, code, Wikipedia, scientific papers. Each one of those 300 billion tokens contributed a tiny nudge to those 175 billion parameters through exactly the same process we just described — cost function, gradient, backpropagation, step. The same algorithm. Just at a mind-boggling scale.
>
> And that's the foundation. You now understand how neural networks learn. Measure how wrong you are (cost function), figure out which direction to adjust (gradient via backpropagation), take a step, and repeat. Let's take a 10-minute break, and when we come back, we'll see how this same idea applies to language.

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

> Welcome back. So far, everything we've covered has used image recognition as the example — feeding in pixel values and classifying digits. But the AI that's dominating headlines right now — ChatGPT, Claude, Gemini — works with *language*, not images.
>
> Here's the good news: everything changes, and nothing changes. The fundamental concepts are identical:
> - Neurons are still just numbers
> - Layers still stack for hierarchical processing
> - Weights are still learned via gradient descent and backpropagation
> - The cost function still measures "how wrong was the prediction"
>
> What changes is the *architecture* — the specific way the layers are arranged — and the *input format*. Instead of pixel brightness values, we're feeding in text. The architecture that makes this work is called the **Transformer**, and the family of models built on it are called GPTs.
>
> Let's break down that acronym. GPT stands for Generative Pre-trained Transformer. **Generative** means it produces new text — it doesn't just classify, it creates. **Pre-trained** means it learned from a massive dataset before you ever touched it — all those billions of gradient descent steps happened before the product launched. **Transformer** is the specific neural network architecture — the arrangement of layers and connections — that made this breakthrough possible. It was introduced in a 2017 paper from Google titled "Attention Is All You Need," which might be the most consequential machine learning paper ever published.
>
> Now let's walk through how text actually gets processed by one of these models.

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

> Neural networks only understand numbers, so the first step is converting text into numbers. The way this works is through **tokenization** — breaking text into chunks called tokens.
>
> Tokens are usually whole words, but not always. Common words like "the" or "and" are single tokens. Longer or less common words get split into pieces. The word "cleverest" might become three tokens: "cle" + "ver" + "est." This is practical — it keeps the vocabulary manageable while still being able to handle any text you throw at it.
>
> GPT-3 has a vocabulary of about 50,257 tokens. Each one has a unique ID number. So when you type a sentence into ChatGPT, the first thing that happens is your text gets chopped into tokens, and each token gets replaced by its ID number. From that point forward, the model never sees your text — it only works with these numbers.
>
> There's also a limit on how many tokens the model can process at once — this is called the **context window**. For GPT-3, it was 2,048 tokens. Modern models like GPT-4 and Claude can handle much more — 100,000 tokens or more. But the concept is the same: there's a fixed window of text the model can "see" at once.

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

> This is where it starts to get really fascinating. Each token — each word or word piece — needs to be represented as a list of numbers the network can work with. That list is called an **embedding**, and it's looked up in a giant table called the embedding matrix.
>
> For GPT-3, each token gets mapped to a list of 12,288 numbers. That's a vector — a point in 12,288-dimensional space. I know that's impossible to visualize — we live in 3 dimensions, and our brains can kind of handle 2D plots. But mathematically, it works the same way.
>
> Now here's what's remarkable: the network *learns* these embeddings during training. Nobody programs them. Nobody tells the model "put 'king' here and 'queen' there." The embeddings start as random numbers, and through billions of gradient descent steps, they self-organize into a structure where words with similar meanings end up close together in this 12,288-dimensional space.
>
> "King" and "queen" end up nearby. "Cat" and "kitten" end up nearby. "Cat" and "refrigerator" are far apart. "Happy" and "joyful" are close; "happy" and "miserable" are far. This structure emerges entirely from the training data — from the model reading billions of sentences and learning which words appear in similar contexts.
>
> The embedding matrix for GPT-3 has about 618 million parameters — 50,257 tokens × 12,288 dimensions. That's over half a billion numbers just to convert words into vectors, and that's less than half a percent of the model's total 175 billion parameters. The real work happens in the layers that come next.

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

> This is probably the most mind-blowing result in all of machine learning, and it's worth spending a minute on.
>
> Take the vector for "woman" and subtract the vector for "man." What you get is a direction in 12,288-dimensional space that represents the concept of gender — or more precisely, the female-vs-male axis. Now take the vector for "king" and add that gender direction. Where do you land? Very close to "queen."
>
> It works for other pairs too. Take "uncle," add the same gender direction, and you land near "aunt." Take "brother," add the direction, and you get "sister." The model has discovered that there's a *direction* in its embedding space that means gender, and it consistently applies across all the word pairs where gender is a distinguishing factor.
>
> And it's not just gender. Take "Italy" minus "Germany" — that gives you a direction that represents something like "Italian-ness vs. German-ness." Now take "Hitler" — a figure associated with Germany — and add the Italy direction. You land near "Mussolini." The model has encoded nationality and historical role as separate directions, and you can combine them.
>
> There's even a direction for plurality. Take "cats" minus "cat" — that difference vector, when you compute dot products with other words, gives higher values for plural nouns than singular ones. And here's a fun detail from the 3Blue1Brown video: if you take that plurality direction and compute its dot product with the words "one," "two," "three," "four," you get *increasing values*. The model has learned that "four" is more plural than "one."
>
> Nobody programmed any of this. Nobody labeled "king" as "male, royal, singular." The model discovered these abstract concepts from reading text — from billions of sentences where "king" appears in similar contexts to "queen" and "ruler" and "monarch," and different contexts from "peasant" or "bicycle." The structure of human language, the relationships between concepts — it's all encoded as geometry in this high-dimensional space.

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

> Here's the core insight about how these models work, and it surprises almost everyone when they first hear it. The entire model — all 175 billion parameters, all of the sophisticated architecture we're about to discuss — is trained to do one thing: predict the next word.
>
> That's it. Given a sequence of text, the model outputs a probability distribution over its entire vocabulary — about 50,000 options — and each one gets a probability. "The capital of France is ___" — "Paris" might get 68%, "a" might get 4%, "the" might get 3%, and so on down the line.
>
> Grant uses a nice example: if the text includes "Harry Potter" and mentions "least favorite" before the word "Professor," the model assigns a very high probability to "Snape." It's learned from the training data that this is the most likely completion.
>
> Text generation works by sampling from this distribution. The model picks one token — weighted by the probabilities — appends it to the text, and then predicts the next one. Over and over. That's how ChatGPT produces entire paragraphs: one word at a time, each word chosen based on everything that came before it.
>
> There's a parameter called **temperature** that controls how random the sampling is. Low temperature makes the model conservative — it almost always picks the highest-probability token. High temperature makes it more creative — it's more willing to pick lower-probability options. At temperature zero, it's completely deterministic — always the top choice. This is why sometimes ChatGPT gives you the same answer twice and sometimes it's different — it depends on the temperature setting.
>
> Now, here's the deep insight: people dismiss this by saying "it's just autocomplete — it's just predicting the next word, like the suggestions on your phone keyboard." And technically, that's true. But think about what it takes to predict well. To complete "The capital of France is ___," you need geography. To complete a legal argument, you need to understand law. To complete a Python function, you need programming logic. To complete "2 + 2 = ___," you need arithmetic. To predict the next word of a Shakespearean sonnet, you need to understand meter, rhyme, and Elizabethan vocabulary.
>
> The simplicity of the goal — just predict the next word — is profoundly deceptive. To do it well at scale, the model has to learn grammar, facts, logic, common sense, style, tone, and reasoning. All from the simple objective of "what word comes next?"

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

> Now let's look at the overall architecture — the Transformer. At a high level, it's remarkably simple.
>
> Tokens come in at the bottom and get converted to embedding vectors — we just covered that. Then those vectors flow upward through a pipeline of alternating blocks: an attention block, then a feed-forward block (also called an MLP or multilayer perceptron), then another attention block, then another feed-forward block. Over and over. GPT-3 repeats this pair 96 times.
>
> The two block types have different jobs:
>
> **Attention blocks** are where the words talk to each other. Each word looks at every other word in the sequence and asks "which of you are relevant to my meaning right now?" It then updates its embedding based on the information it gathers. We'll go deep on this in the next block.
>
> **Feed-forward blocks** (MLPs) process each word independently — no inter-word communication. This is where factual knowledge gets injected. "Is this vector encoding Michael Jordan? If so, add information about basketball." We'll touch on this at the end.
>
> After 96 rounds of "talk to your neighbors" (attention) and "add knowledge" (MLP), the final vector for the last token in the sequence passes through an **unembedding matrix** — essentially the reverse of the embedding step. This maps the 12,288-dimensional vector back into a list of ~50,000 scores, one per vocabulary token. Then the softmax function turns those scores into probabilities, and the model picks the next word.
>
> That unembedding matrix is another 618 million parameters. So the embedding and unembedding together account for about 1.2 billion of GPT-3's 175 billion parameters. The remaining 174 billion are in the 96 layers of attention and feed-forward blocks. That's where the real intelligence lives. Let's go look at attention — the mechanism that changed everything.

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

> Here's the problem that attention solves. When a word first enters the network, it gets its embedding from the lookup table — and it's the same embedding every time, regardless of context. The word "mole" gets the same 12,288 numbers whether it appears in "American shrew mole" (an animal), "one mole of carbon dioxide" (a chemistry unit — 6.022 × 10²³ particles), or "take a biopsy of the mole" (a skin growth). Three completely different meanings, but the initial embedding is identical.
>
> Grant uses another great example: "Eiffel tower" vs. "miniature tower." The word "tower" starts with the same embedding in both cases. But "Eiffel tower" should evoke Paris, wrought iron, 1,000 feet tall. "Miniature tower" is tiny. The context completely changes what the word means.
>
> Or think about the word "quill." In a Harry Potter context, it's a writing instrument. In a nature documentary, it's a hedgehog spine. Same word, same initial embedding, completely different meaning.
>
> The network needs a mechanism for surrounding words to influence each other's meanings — to let "Eiffel" reach over and update "tower" with information about iron, Paris, and enormous height. That mechanism is **attention**, and it's arguably the single most important innovation in modern AI. It's the key idea in that 2017 paper "Attention Is All You Need," and it's what makes Transformers work.

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

> Let me walk you through how attention actually works. I'm going to use a simplified example from Grant's video: the sentence "A fluffy blue creature roamed the verdant forest."
>
> The goal of attention is to let adjectives like "fluffy" and "blue" update the meaning of the noun they describe — "creature" — so that by the time we're done, the vector for "creature" doesn't just mean "some generic creature" but specifically "a fluffy, blue creature."
>
> Here's the mechanism. Three learned matrices are involved — **query, key, and value** — each producing a different vector for each word.
>
> **Queries** represent what a word is *looking for*. Think of nouns generating a query that says "hey, are there any adjectives sitting in front of me?" The query matrix — a set of learned weights — transforms each word's embedding into this question vector.
>
> **Keys** represent what a word *has to offer*. Adjectives like "fluffy" and "blue" generate keys that essentially say "I'm a descriptor! I have information about properties!" The key matrix transforms embeddings into these answer vectors.
>
> **The match:** For every pair of words, the model computes a dot product between the query of one word and the key of another. Remember from the embedding discussion — a dot product measures alignment. When a query and key are well-aligned, the dot product is large, meaning "these two words are relevant to each other." When they're not aligned, the dot product is small or negative.
>
> So "creature"'s query (looking for adjectives) will have a high dot product with "fluffy"'s key (offering adjective information) and "blue"'s key — but a low dot product with "roamed"'s key or "the"'s key.
>
> These raw scores then go through softmax — the same function from earlier — to normalize them into a distribution that sums to 1. Now each word has a set of attention weights: "I should pay 40% attention to 'fluffy,' 35% to 'blue,' 10% to 'A,' and so on."
>
> **Values** are the third piece. The value matrix transforms each word into the *actual information* to contribute. "Fluffy"'s value vector encodes the specific meaning to add — not "I'm an adjective" generically, but the particular semantic content of fluffiness. This value vector gets multiplied by the attention weight and added to "creature"'s embedding.
>
> The result: "creature"'s embedding gets nudged in a direction that incorporates fluffiness and blueness. It's been updated by context.
>
> Think of it like a networking event. Everyone walks in wearing two badges: a name tag saying what they know (their key), and a lanyard card saying what they're looking for (their query). You scan the room, find the people whose name tags match your needs, and spend the most time talking to them. By the end of the event, your understanding has been enriched by the people you connected with. That's attention.

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

> What we just described is one attention head — one set of query, key, and value matrices learning one type of relationship. But language has many types of relationships: grammar, pronoun references, semantic similarity, sentiment, logical connections, temporal ordering...
>
> So the model doesn't run attention once — it runs it 96 times in parallel, with 96 completely separate sets of query, key, and value matrices. Each one is called an **attention head**, and each one learns to look for a different type of relationship.
>
> One head might learn to connect subjects with their verbs. Another might learn to connect pronouns with the nouns they refer to — figuring out that "it" in "The animal didn't cross the street because it was too tired" refers to "animal," not "street." Another might track whether the text is talking about Harry Potter or Prince Harry — looking at words like "wizard" vs. "Queen" and "Sussex" to disambiguate.
>
> All 96 heads produce their own suggested updates to each word's embedding, and those updates all get summed together. So each word gets enriched from 96 different perspectives simultaneously.
>
> And this happens at every one of GPT-3's 96 layers. So the total number of attention operations is 96 heads × 96 layers = 9,216. Each word gets analyzed from over 9,000 different perspectives as it flows through the network. About 58 billion of GPT-3's 175 billion parameters are devoted to attention alone — roughly a third of the entire model.
>
> One important technical detail: **masking**. When the model is training, it processes entire sequences at once for efficiency. But it can't let the word at position 50 attend to the word at position 51 — that would be letting it see the future, seeing the answer before making the prediction. So all attention scores between a word and any *later* word are forced to zero. This is done by setting those scores to negative infinity before the softmax, which turns them into zeros. The attention pattern ends up looking like a triangle — each word can only attend to words before it.
>
> Grant makes an important point about why attention succeeded: it's not just that it's a clever mechanism. A big part of its success is that it's "extremely parallelizable" — you can run all the dot products on a GPU simultaneously. This is what enabled the massive scaling that has driven recent AI progress. The mechanism is good, but the ability to make it enormous is what made the breakthrough.

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

> Let me paint a picture of the transformation that attention and the subsequent layers produce.
>
> When the word "king" enters the network, its embedding is generic. It means "king" — royalty, ruler, monarchy — but no specific king in any specific context. It's a dictionary definition.
>
> After flowing through 96 layers of attention and feed-forward processing, that same vector might encode something like: "a fictional Scottish king who murdered his predecessor to seize the throne, described in a play written around 1606, and the text is currently building toward the psychological consequences of that act." All of that is packed into 12,288 numbers — the same vector that started as just "king."
>
> Grant gives another powerful example: imagine a mystery novel. The very last word of the text so far is "was" — as in "Therefore, the murderer was..." That word "was" started as a generic past-tense verb. But by the time it reaches the final layer, its embedding has been updated by 96 rounds of attention — pulling in information from every relevant clue, every character mention, every red herring in the entire context window. By the final layer, "was" is no longer just a verb — it's a compressed representation of the entire mystery, encoding which character the evidence points to. And that enriched vector is what gets unembedded into the probability distribution that (hopefully) assigns the highest probability to the correct suspect's name.
>
> That's the power of attention. It takes flat, context-free word representations and transforms them into deep, contextually-rich vectors that encode meaning, relationships, and knowledge accumulated over many layers of processing.

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

> We've spent most of our time on attention, and rightfully so — it's the signature innovation. But there's a whole other half of each Transformer block that we've been glossing over: the feed-forward layers, also called MLPs — multilayer perceptrons. And they're actually where the majority of the model's parameters live.
>
> Remember, each layer of the Transformer has two parts: attention (where words look at each other) and MLP (where each word gets processed independently). Attention is about relationships. MLPs are about knowledge.
>
> Here's how an MLP works, using Grant's example. Imagine the vector flowing through the network currently encodes "Michael Jordan." The MLP needs to recognize this and add relevant facts — basketball, Chicago Bulls, jersey number 23.
>
> **Step 1: Up-projection.** The vector gets multiplied by a huge matrix with about 50,000 rows. Each row is essentially a question, encoded as a direction in embedding space. One row might represent the direction "Michael + Jordan" — it's asking "does this vector point in the Michael direction AND the Jordan direction?" The dot product between the input vector and this row gives a high number only if both conditions are met. Then a bias of -1 is subtracted — meaning the match must be strong enough that both names are present, not just one.
>
> **Step 2: ReLU gate.** The result goes through ReLU — if it's positive, the "neuron" activates. If it's negative or zero, the neuron stays at zero. This is essentially a yes/no switch. And because of how the bias works, it acts like an AND gate: it fires for "Michael Jordan" but not for just "Michael" or just "Jordan." Pretty clever.
>
> **Step 3: Down-projection.** Each activated neuron corresponds to a column in a second matrix. That column encodes the knowledge to inject — a direction in embedding space that represents "basketball" or "Chicago Bulls." Active neurons contribute their knowledge; inactive neurons contribute nothing (they're multiplied by zero). A single neuron's column can encode *multiple* associated facts simultaneously — not just basketball, but also Chicago Bulls, number 23, the dunk from the free throw line.
>
> **Step 4:** The result gets added back to the original vector. Now the vector that used to just encode "Michael Jordan" also encodes "plays basketball, Chicago Bulls, #23." The model has injected factual knowledge.
>
> The scale is staggering. Each MLP block in GPT-3 has about 1.2 billion parameters. With 96 layers, that's roughly 116 billion parameters devoted to MLPs — about two-thirds of the entire model. Attention gets all the headlines, but the majority of the model is actually these knowledge-storage layers.
>
> One last mind-bending detail from Grant's video: you might think each neuron cleanly represents one concept, like "Michael Jordan." In reality, individual neurons rarely represent single clean features. Instead, the model uses something called **superposition** — features overlap and share the same neurons, like multiple radio stations broadcasting on overlapping frequencies. A mathematical result called the Johnson-Lindenstrauss lemma shows that in 12,288-dimensional space, you can pack more than 40 billion nearly-independent directions — exponentially more than the number of dimensions. So the model can store vastly more concepts than it has neurons. As Grant puts it: "A space that has 10 times as many dimensions can store way, way more than 10 times as many independent ideas." This is why bigger models are so much more capable — more dimensions means exponentially more room for knowledge.

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

> Let's take a step back and see how far we've come in two hours.
>
> We started with the simplest possible building block — a neuron, which is just a number between 0 and 1. We stacked neurons into layers, with each layer handling a different level of abstraction — edges, then patterns, then answers. We connected them with weights — volume knobs that control how much influence each neuron has on the next — and biases that set minimum thresholds.
>
> Then we learned how those weights and biases get set: through the cost function (measuring how wrong the network is), gradient descent (rolling downhill to reduce that error), and backpropagation (tracing the error backward through the network to figure out which knobs to turn and by how much).
>
> In the second hour, we took those exact same ideas and applied them to language. Words become tokens, tokens become embeddings — lists of numbers where similar words are close together and directions encode meaning. We learned about attention — the mechanism that lets each word look at every other word and update its meaning based on context — and we saw how it runs 96 heads in parallel across 96 layers for over 9,000 total attention passes. And we closed with MLPs — the layers that store factual knowledge, making up two-thirds of the entire model.
>
> Nine concepts. That's the entire foundation of modern AI. Everything from ChatGPT to Claude to image generators to self-driving cars is built on these nine ideas.

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

> Everything we've covered so far is about the *brain* — the AI model itself. How it's built, how it learns, how it processes language. But a brain sitting in a jar can't *do* anything in the real world. It can answer questions and generate text, but it can't configure a firewall, query a database, check inventory levels, or file a report.
>
> This afternoon, we're going to give that brain a body. Three things:
>
> First, a **workflow engine** — that's LangGraph. It's the plumbing that defines what steps exist, what order they run in, and what data flows between them. Think of it as the organizational chart and the process flow.
>
> Second, **tools** — Python functions that the AI can discover and call. Check a version number. Create a configuration. Query a threat intel database. These are the hands — the specific actions the brain can take in the real world.
>
> Third, a **decision loop** — the ability for the AI to look at a situation, decide what to do, execute it, look at the result, and decide whether it needs to do more. This is what people mean when they say "AI agent" — it's not just responding, it's *reasoning, acting, observing, and adapting*.
>
> Now when we get to the workshop and I say "the LLM decides what to do next," you know what's actually happening under the hood. The text of the situation is tokenized, embedded into 12,288-dimensional vectors, run through 96 layers of attention and MLPs, and the model predicts which tool to call next based on all of its training. Attention is weighing which parts of the context matter most. The MLPs are injecting relevant knowledge. And gradient descent on 300 billion training examples is what tuned all 175 billion parameters to make those decisions well.
>
> You understand the brain. Now let's give it a body.

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

> That's a wrap on Session 0. Let me leave you with this: you now have a genuine understanding of how AI works — not just what it does, but *how it does it*. You know what a neuron is, how layers decompose hard problems, how gradient descent and backpropagation train the network, how tokens and embeddings represent language, how attention enables context, and how MLPs store knowledge.
>
> When someone at a dinner party says "AI is going to take over the world," you can say "Well, it's actually a function that predicts the next word by computing dot products between learned query and key vectors across 96 attention heads, enriched by feed-forward layers that inject factual knowledge, all trained via stochastic gradient descent on 300 billion tokens of text." That should clear the room pretty quickly. [pause for laugh]
>
> But seriously — this foundation is going to make everything this afternoon click. When we start building AI agents, you'll understand what's happening under the hood at every step. Let's take 15 minutes, grab some coffee, and then we'll start building.

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
