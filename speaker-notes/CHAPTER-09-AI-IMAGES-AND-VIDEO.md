# Chapter 9 — But How Do AI Images and Videos Actually Work?

## Presentation Guide

**Duration:** 45–55 minutes
**Audience:** Non-technical; mixed backgrounds (National Guard unit)
**Goal:** Build intuition for how diffusion models generate images and videos from text prompts — covering CLIP embeddings, the diffusion process, and classifier-free guidance. Connect to physics (Brownian motion) and geometric intuitions from earlier sessions.
**Source material:** Concepts and examples adapted from [3Blue1Brown's Deep Learning series — "But how do AI images and videos actually work?"](https://www.youtube.com/watch?v=iv-5mZ_9CPY) (guest video by Welch Labs)

---

## Agenda at a Glance

| Time | Slide Block | Duration |
|------|-------------|----------|
| 0:00 | **The big picture: text → images** | 3 min |
| 0:03 | **Hands-on demo: WAN 2.1** | 4 min |
| 0:07 | **Three parts: CLIP, Diffusion, Guidance** | 3 min |
| 0:10 | **CLIP: shared text/image space** | 5 min |
| 0:15 | **CLIP embedding space properties** | 4 min |
| 0:19 | **The diffusion process: adding and removing noise** | 5 min |
| 0:24 | **2D toy dataset intuition** | 4 min |
| 0:28 | **Score function and vector fields** | 4 min |
| 0:32 | **Why adding noise during generation helps** | 4 min |
| 0:36 | **DDIM: deterministic generation** | 3 min |
| 0:39 | **Classifier-free guidance: steering with text** | 5 min |
| 0:44 | **Negative prompts** | 3 min |
| 0:47 | **The remarkable thing: it all fits together** | 3 min |

**Total: ~50 minutes (13 slides)**

---

## The Big Picture

### SLIDE 01 — Text in, images out

**Headline:** AI can now turn text prompts into images and videos — and the process is connected to 19th-century physics.

**Body:**
- Type a description → get a photorealistic image or video that never existed before
- "An astronaut riding a horse on Mars" → the model generates it from scratch
- The core technique: **diffusion** — systematically removing noise to reveal an image
- The physics connection: diffusion models reverse **Brownian motion** — the random jittering of particles that Einstein described in 1905
- Three ingredients make it work: **CLIP** (understanding text and images together), **Diffusion** (noise removal), and **Guidance** (steering with text)

**Visual concept:** A text prompt "An astronaut planting a flag on the moon" → an arrow → a generated image of that scene. Below, a timeline showing: random noise → progressively clearer image → final result, labeled "Diffusion: noise removal in reverse."

**Speaker notes:**

- Generates images/videos from text — not retrieving from a database, creating entirely new content
- Core process = diffusion, connected to real physics (Brownian motion, Einstein 1905) run in reverse
- Three components: CLIP (bridges text and images), Diffusion (noise-removal engine), Guidance (text steers generation)
- Same geometric/mathematical intuitions from earlier sessions apply here

---

### SLIDE 02 — Hands-on demo: WAN 2.1

**Headline:** Open-source models let anyone generate AI videos — starting from pure random noise.

**Body:**
- **WAN 2.1** — an open-source video generation model anyone can download and run
- Demo prompts: "astronaut holding a flag," "astronaut holding a laptop," "astronaut in a meeting"
- The generation process is visible: starts from a screen of random static/noise
- The same transformer architecture passes over the noise repeatedly, each pass removing some noise
- After enough passes, a coherent video emerges from what started as pure randomness

**Visual concept:** A filmstrip showing 6 frames from the generation process: Frame 1 = pure random noise, Frame 2 = vague shapes emerging, Frame 3 = rough silhouette, Frame 4 = recognizable astronaut form, Frame 5 = detailed scene, Frame 6 = final polished video frame.

**Speaker notes:**

- WAN 2.1: open-source, anyone with a GPU can download and run it
- Demo shows generation in real time: pure static → shapes emerge → objects form → details sharpen → final video
- Not "drawing" from blank canvas — "revealing" an image hidden in noise (sculptor removing marble)
- Same transformer architecture (attention + MLP) from earlier sessions, adapted for images/video

---

## Three Parts

### SLIDE 03 — The three-part system

**Headline:** CLIP + Diffusion + Guidance = text-to-image generation.

**Body:**
- **CLIP** (Contrastive Language-Image Pre-training):
  - Creates a shared space where text and images can be directly compared
  - "Understands" both words and pictures using the same set of numbers
- **Diffusion:**
  - The engine that generates images by systematically removing noise
  - Trained by learning to reverse a noise-addition process
- **Guidance:**
  - The steering mechanism that uses text to control what the diffusion model generates
  - Without guidance, the model generates generic images; with it, the model follows your prompt

**Visual concept:** Three interlocking puzzle pieces or gears: CLIP (blue, labeled "shared understanding"), Diffusion (green, labeled "noise removal engine"), Guidance (orange, labeled "text steering"). Arrows showing how they connect: CLIP feeds into Guidance, which steers Diffusion.

**Speaker notes:**

- CLIP = translator between text and images; Diffusion = generation engine; Guidance = steering wheel
- Covering in order: CLIP first (foundation), then Diffusion (noise removal), then Guidance (prompt control)
- Each was a separate research breakthrough — the magic is how they fit together

---

## CLIP: Shared Text/Image Space

### SLIDE 04 — Two models, one shared space

**Headline:** CLIP trains a text model and an image model to produce vectors in the same 512-dimensional space.

**Body:**
- CLIP uses **two separate neural networks**: one for text, one for images
- Both output a **512-dimensional vector** (a list of 512 numbers)
- Trained on **400 million image-caption pairs** scraped from the internet
- **Contrastive training:**
  - Matching pairs (image + its correct caption) → vectors pushed **close together**
  - Non-matching pairs (image + wrong caption) → vectors pushed **far apart**
- Similarity measured by **cosine similarity** — the angle between two vectors
  - Vectors pointing the same direction = high similarity
  - Vectors pointing in different directions = low similarity

**Visual concept:** Two neural networks side by side: "Text Encoder" (taking in "a photo of a dog") and "Image Encoder" (taking in a photo of a dog). Both output arrows pointing to the same region in a shared space, labeled "512-dim vector." Below: a grid showing matching pairs (green checkmarks, vectors aligned) and non-matching pairs (red X's, vectors divergent).

**Key insight card:**
After training, you can compare ANY text to ANY image by measuring the angle between their vectors. A photo of a sunset and the words "beautiful sunset" will point in similar directions — even though one is pixels and the other is words.

**Speaker notes:**

- OpenAI trained CLIP on 400M image-caption pairs scraped from the internet
- Two networks: text encoder → 512-dim vector; image encoder → 512-dim vector; same shared space
- Contrastive training: matching pair → push vectors closer; non-matching → push apart; repeat 400M times
- After training: same-content text and images point in similar directions; cosine similarity measures the angle
- Without CLIP, no way to connect "astronaut" (word) to astronaut images (pixels)

---

### SLIDE 05 — Mathematical operations on concepts

**Headline:** In CLIP's embedding space, you can do math on pure concepts — and it works.

**Body:**
- Take an image of a person **wearing a hat** and an image of the **same person without a hat**
- Subtract one vector from the other → you get a **difference vector**
- Find the closest text to that difference vector → the answer is **"hat"**
- The model has isolated the pure concept of "hat-ness" as a direction in 512-dimensional space
- Just like word embeddings: "king" - "man" + "woman" ≈ "queen"
- But now it works **across modalities** — mixing images and text in the same mathematical space

**Visual concept:** Two photos side by side: person with hat, person without hat. An arrow showing subtraction of their vectors. The resulting difference vector points toward the text embedding of the word "hat." Include the equation: image(hat) - image(no hat) ≈ text("hat").

**Speaker notes:**

- image(person with hat) - image(person without hat) → difference vector → nearest text = "hat"
- Model learned a general "hat-ness" direction — not tied to any specific hat or person
- Same as "king - man + woman ≈ queen" but now works ACROSS modalities (images and text in same space)
- Why it matters: any concept = a direction, any prompt = a vector → mathematical target for diffusion to aim at

---

## The Diffusion Process

### SLIDE 06 — Forward process: systematically destroying an image with noise

**Headline:** Add noise to an image step by step until it's completely destroyed — then train a model to reverse it.

**Body:**
- **Forward process (training):**
  - Take a clean training image
  - Add a small amount of random noise → slightly grainy image
  - Add more noise → grainier
  - Keep adding → eventually pure random static
- **The model's job:** Learn to predict the **total noise** that was added
  - NOT step-by-step denoising — predict ALL the noise in one shot
  - Given a noisy image, output what the noise looks like
  - Subtract the predicted noise → recover the original image
- **Key insight:** The model doesn't need to reverse one step at a time — it learns to skip all intermediate steps

**Visual concept:** Top row (forward): clean photo → slightly noisy → more noisy → very noisy → pure static, with "add noise" arrows between each. Bottom: a neural network that takes in the noisy image and outputs the predicted noise pattern, which gets subtracted to recover the clean image.

**Analogy card:**
Imagine someone slowly stirring mud into clear water, taking photos at each stage. Now train a model to look at any of those muddy-water photos and figure out exactly where all the mud particles are — so you can remove them all at once and get back to clear water.

**Speaker notes:**

- Training: take clean image → add noise in steps until destroyed → pure random noise
- Model learns to predict TOTAL noise in one shot (not step-by-step) → subtract predicted noise → recover image
- Key subtlety: predicts ALL noise at once, not one step — much more efficient
- Generation: start from pure random noise → predict/remove noise in several steps → coherent image emerges

---

### SLIDE 07 — 2D toy dataset: images as points in space

**Headline:** Think of every possible image as a point in a vast space. Real images cluster in specific regions. Noise scatters them randomly.

**Body:**
- Every image can be represented as a point in high-dimensional space (one dimension per pixel)
- Real, natural images don't fill the whole space — they cluster in specific regions
- **2D analogy:** Imagine all real images form a spiral pattern in a simple 2D space
- **Adding noise** = giving each point a random push — a random walk (Brownian motion)
  - Small noise: points jitter near their original positions — spiral still visible
  - More noise: points spread out — spiral starts to blur
  - Lots of noise: points scattered uniformly — no structure left
- **This IS Brownian motion** — the random jittering of particles that Einstein described
- The diffusion model learns to reverse these random walks — push scattered points back toward the spiral

**Visual concept:** A 2D spiral of dots (representing "real images"). Three stages: (1) dots tightly on the spiral, (2) dots slightly scattered but spiral shape still visible, (3) dots fully scattered into a uniform cloud. Arrows on the scattered dots showing the "reverse" direction — pointing back toward the spiral.

**Speaker notes:**

- Every image = a point in high-dimensional space (1-megapixel = million dimensions); use 2D spiral as intuition
- Real images cluster in tiny fraction of the space — most random pixel combos are just static
- Adding noise = random walks (Brownian motion): small nudges → stay near spiral; big nudges → scatter everywhere
- Diffusion model reverses the walks: given a scattered point, which direction leads back to real data?

---

### SLIDE 08 — The score function: a vector field pointing home

**Headline:** The model learns a vector field — at every point in the space, an arrow pointing back toward real data.

**Body:**
- At every point in the noisy space, the model learns a **direction** that points back toward the original data distribution
- This is called the **score function** — a vector field covering the entire space
- **Time conditioning is essential:**
  - At large noise levels (large t): arrows point toward the coarse structure — general shape, overall composition
  - At small noise levels (small t): arrows point toward fine details — textures, edges, specific features
  - The model needs to know HOW noisy the input is to give the right directions
- Early in generation: big corrections to get the rough structure right
- Late in generation: small corrections to refine details

**Visual concept:** A 2D field of arrows (vector field) with the spiral faintly visible underneath. Arrows far from the spiral are long and point toward the general center. Arrows near the spiral are short and point toward the exact nearest position on the spiral. Side panel: two arrow fields labeled "Large t (coarse)" with big sweeping arrows and "Small t (fine)" with tiny precise arrows.

**Key insight card:**
Time conditioning is what makes diffusion work at all. Without knowing the noise level, the model can't decide whether to make big structural corrections or tiny detail refinements. It needs the clock to know what kind of fix to apply.

**Speaker notes:**

- Model learns a vector field (score function): at every point, an arrow pointing back toward real data
- Score function = gradient of log probability = "direction toward the good stuff"
- Time conditioning is critical: model takes noisy data + time step t (tells it HOW noisy)
- Large t → focus on big picture (face? landscape?); small t → focus on fine details (edges, textures)
- Generation goes large-t to small-t: coarse structure first → medium features → fine details last

---

### SLIDE 09 — Why noise during generation is necessary

**Headline:** Without adding noise during generation, everything converges to the average — the model produces blurry, generic images.

**Body:**
- Problem: the model learns the **mean** (average) of the data distribution at each point
- If you just follow the arrows without any randomness, all paths converge to the same average image
- Result: blurry, generic, boring — the "average face," the "average landscape"
- **Solution:** Add small amounts of random noise during generation
  - Noise prevents convergence to the mean
  - Allows the model to sample from the FULL distribution of possible images
  - Different noise → different images from the same prompt
- This is why you get different results each time you generate — the random noise is different

**Visual concept:** Two generation paths side by side: Left path (no noise) shows arrows converging to a single blurry "average" image in the center. Right path (with noise) shows paths that wander and end up at different sharp, detailed images distributed around the spiral.

**Speaker notes:**

- Model learns AVERAGE direction at each point — following it all the way → blurry generic "average image"
- Adding noise during generation = random kicks that prevent convergence to the mean
- Different random kicks → different final images; same prompt produces different results each time
- Math: stochastic differential equations; noise ensures sampling from full distribution, not just the center

---

## DDIM: Deterministic Generation

### SLIDE 10 — DDIM: fewer steps, same quality

**Headline:** DDIM removes the randomness and uses differential equations for deterministic, faster generation.

**Body:**
- **DDPM** (original): Uses random noise at each step — stochastic, requires many steps (~1000)
- **DDIM** (improved): Uses **ordinary differential equations** (ODEs) — no random noise needed
  - Deterministic: same starting noise always produces the same image
  - Fewer steps needed: can produce good results in 20-50 steps instead of 1000
  - Same quality distribution as DDPM
- The vector field the model learned can be followed like a smooth flow instead of a noisy random walk
- Trade-off: no diversity from noise (but you can get diversity from different starting noise)

**Visual concept:** Two paths from the same noisy starting point: Left (DDPM) = jagged, zigzag path with random jitters at each step, taking many steps. Right (DDIM) = smooth, curved path following the flow field, reaching the same quality result in fewer steps.

**Speaker notes:**

- DDPM (original): random noise each step, ~1000 steps needed, slow
- DDIM (improved): smooth ODEs instead of random walk, deterministic, 20-50 steps instead of 1000
- Same starting noise = same final image; diversity comes from choosing different starting noise
- Intuition: smooth flow vs. noisy walk — huge speedup, same quality

---

## Classifier-Free Guidance

### SLIDE 11 — Steering generation with text prompts

**Headline:** Compare the model's output WITH your prompt vs. WITHOUT it — amplify the difference. That's guidance.

**Body:**
- Run the diffusion model **twice** at each step:
  1. **Conditioned** run: "remove noise, given the prompt 'a tree in autumn'"
  2. **Unconditioned** run: "remove noise, with no text guidance"
- Subtract the unconditioned direction from the conditioned direction → get the **prompt-specific component**
- **Amplify** this difference by a scaling factor (called **alpha** or **guidance scale**)
  - Alpha = 1: follow the prompt normally
  - Alpha > 1: exaggerate the prompt's influence — more literal, more dramatic
  - Alpha very high: over-saturated, distorted (but very prompt-adherent)
- Example: with "a tree" prompt and increasing alpha, the tree literally GROWS — becomes more and more tree-like

**Visual concept:** Two arrows from the same point: one labeled "conditioned" (pointing toward a tree image), one labeled "unconditioned" (pointing toward a generic/average image). The difference arrow is shown, then amplified. Three result images at alpha=1 (normal tree), alpha=3 (vivid, lush tree), alpha=10 (exaggerated, oversaturated tree). Show the tree literally growing as alpha increases.

**Key insight card:**
Classifier-free guidance asks: "What's the difference between what you'd generate WITH this prompt vs. WITHOUT it?" Then it amplifies that difference. It isolates and boosts the effect of your words.

**Speaker notes:**

- Two runs per step: conditioned (with prompt) vs. unconditioned (no prompt) — difference = prompt's contribution
- Subtract unconditioned from conditioned → isolate prompt's effect → amplify by alpha (guidance scale)
- Higher alpha = more prompt influence (tree literally grows); too high = oversaturated/distorted
- Computationally 2x (two passes per step) but dramatically better results

---

### SLIDE 12 — Negative prompts: steering AWAY from what you don't want

**Headline:** Negative prompts subtract unwanted features — telling the model what to avoid.

**Body:**
- Same guidance mechanism, but used in reverse
- **Negative prompt:** Features you DON'T want in the output
  - "extra fingers," "blurry," "walking backwards," "low quality"
- The model computes the direction for the negative prompt and **subtracts** it
- Steers the diffusion process AWAY from those unwanted features
- WAN 2.1 example: negative prompts removing extra fingers and backward walking from generated videos
- Combining positive and negative prompts gives fine-grained control over the output

**Visual concept:** A vector diagram showing three arrows from a point: "conditioned direction" (toward desired image), "negative prompt direction" (toward undesired features like extra fingers), and the "final direction" (the conditioned direction with the negative subtracted — curving away from the bad features). Result images: without negative prompt (astronaut with 6 fingers) vs. with negative prompt (astronaut with 5 fingers).

**Speaker notes:**

- Same guidance math in reverse: positive = "steer toward," negative = "steer away" (subtract direction)
- Common negatives: "extra fingers," "walking backwards," "blurry, low quality, deformed"
- Compute negative prompt direction → subtract from generation direction → model actively avoids those features
- Real workflow: positive prompt (what you want) + negative prompt (what you don't) + guidance scale + multiple seeds

---

## The Remarkable Thing

### SLIDE 13 — It all fits together: CLIP + Diffusion + Guidance

**Headline:** Three independently developed ideas snap together like puzzle pieces — and high-dimensional geometric intuitions hold all the way through.

**Body:**
- **CLIP** creates a shared space where text and images coexist as vectors — concepts become directions
- **Diffusion** learns to reverse noise — following a vector field from randomness back to real images
- **Classifier-free guidance** uses CLIP's text understanding to steer diffusion's vector field toward your prompt
- The geometric intuitions from embedding spaces (directions = concepts, dot products = similarity) carry over perfectly
- The same high-dimensional math that lets LLMs store facts also lets diffusion models generate images
- Welch Labs' key phrase: *"All you need is language."*
  - Once you can represent concepts as directions in a shared space, generation follows naturally
  - The bridge between words and images is geometry

**Visual concept:** A unified diagram showing the full pipeline: Text prompt → CLIP text encoder → 512-dim vector → Guidance mechanism → steers → Diffusion model (noise → image through vector field) → final generated image. All components connected with arrows, and the shared 512-dim space highlighted as the linchpin connecting everything.

**Key insight card:**
These weren't designed as one unified system. CLIP, diffusion, and guidance were separate research breakthroughs. The fact that they snap together so cleanly suggests something deep about the geometry of meaning in high-dimensional spaces.

**Speaker notes:**

- Three independent breakthroughs that snap together: CLIP (shared space), Diffusion (reverse noise), Guidance (steer with text)
- Unifying thread = high-dimensional geometry: concepts are directions, dot products measure similarity, vector fields guide generation
- "All you need is language" — once concepts become vectors: search, generate, and edit images with text
- Same math as LLM embedding spaces from earlier sessions, just applied to a new domain

---

## Slide Count Summary

| Block | Slides | Duration |
|-------|--------|----------|
| The big picture | 1 (slide 01) | 3 min |
| Hands-on demo | 1 (slide 02) | 4 min |
| Three-part system | 1 (slide 03) | 3 min |
| CLIP | 2 (slides 04-05) | 9 min |
| Diffusion process | 2 (slides 06-07) | 9 min |
| Score function | 1 (slide 08) | 4 min |
| Why noise helps | 1 (slide 09) | 4 min |
| DDIM | 1 (slide 10) | 3 min |
| Classifier-free guidance | 1 (slide 11) | 5 min |
| Negative prompts | 1 (slide 12) | 3 min |
| The remarkable thing | 1 (slide 13) | 3 min |
| **Total** | **13 slides** | **~50 minutes** |

---

## Attribution

Concepts, examples, and explanatory frameworks adapted from the [3Blue1Brown Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Specific content drawn from:
- ["But how do AI images and videos actually work?"](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — guest video by Welch Labs (2024)

Additional references:
- OpenAI CLIP: Contrastive Language-Image Pre-training (Radford et al., 2021)
- DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- DDIM: Denoising Diffusion Implicit Models (Song et al., 2021)
- Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
- WAN 2.1 open-source video generation model
