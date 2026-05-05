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

- Start with the "wow" factor — this technology generates images and videos from text descriptions
  - Images that never existed, scenes that were never photographed
  - Not retrieving from a database — creating entirely new visual content
- The underlying process is called **diffusion**
  - Connected to real physics — Brownian motion
  - Einstein's 1905 paper described how pollen grains jitter randomly in water
  - Diffusion models essentially run this process in reverse
- Three key components we'll cover:
  - CLIP — bridges the gap between text and images
  - Diffusion — the noise-removal engine
  - Guidance — how text steers the generation
- Same geometric and mathematical intuitions from earlier sessions apply here

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

- WAN 2.1 is a real, open-source model — not behind a paywall
  - Anyone with a powerful enough GPU can download and run it
  - Makes the technology tangible and inspectable
- The demo shows the generation process in real time
  - Starts from complete random noise — pure static
  - Each pass through the transformer removes some noise
  - Gradually: shapes emerge → objects form → details sharpen → final video
- Key observation: the model isn't "drawing" the image from a blank canvas
  - It's "revealing" an image that's hidden inside random noise
  - Like a sculptor removing marble to reveal the statue inside
- The transformer architecture from our earlier sessions is doing the heavy lifting
  - Same attention + MLP structure, adapted for images/video instead of text

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

- Three components, each doing a distinct job
  - CLIP = the translator between text and images
  - Diffusion = the image-generation engine
  - Guidance = the steering wheel that makes the engine follow your prompt
- We'll cover each one in order
  - CLIP first — it's the foundation that makes text-to-image possible
  - Then Diffusion — how noise removal creates images
  - Then Guidance — how your prompt actually controls the output
- Each component was a separate research breakthrough
  - The magic is in how they fit together

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

- CLIP was developed by OpenAI — trained on 400 million image-caption pairs
  - Massive dataset scraped from the internet
  - Each example: an image paired with a text description
- Two separate neural networks:
  - Text encoder: takes in words → outputs a 512-dimensional vector
  - Image encoder: takes in an image → outputs a 512-dimensional vector
  - Both vectors live in the SAME 512-dimensional space
- Contrastive training — the key innovation:
  - Show the model a photo of a dog and the caption "a photo of a dog"
  - Push their vectors CLOSER together in the shared space
  - Show the model the same photo with the caption "a red sports car"
  - Push their vectors FARTHER apart
  - Repeat 400 million times
- After training:
  - Images and text descriptions of the same thing point in similar directions
  - Cosine similarity = measuring the angle between vectors
  - Small angle = similar, large angle = different
- This is what bridges the gap between text and images
  - Without CLIP, there's no way to connect "astronaut" (a word) to astronaut images (pixels)

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

- Same concept arithmetic from word embeddings, but now spanning images AND text
  - Take a photo of someone in a hat
  - Take a photo of the same person without a hat
  - Subtract the image vectors
  - The difference points in a direction — find the nearest text
  - The nearest text is "hat"
- The model has learned a pure concept direction for "hat"
  - Not tied to any specific hat, any specific person, any specific photo
  - A general "hat-ness" direction in the shared space
- This is the same kind of geometric structure we saw with word embeddings
  - "King - man + woman ≈ queen" — directions encoding concepts
  - But CLIP extends this across modalities
  - Text and images are in the SAME space with the SAME geometric structure
- Why this matters for image generation:
  - If you can represent ANY concept as a direction in this space...
  - And you can represent ANY text prompt as a vector in this space...
  - Then you have a mathematical target for the diffusion model to aim at

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

- The training process for diffusion models is elegantly simple
  - Start with a clean image from the training set
  - Add random noise in steps — each step adds a little more
  - Eventually the image is completely destroyed — pure random noise
- The model learns to reverse this destruction
  - Given a noisy image, predict what the TOTAL noise looks like
  - This is a key subtlety — NOT predicting one step of noise
  - It predicts ALL the noise that was added, in one shot
  - Subtract the predicted noise → get back to the clean image
- Why predict total noise instead of single steps?
  - Much more efficient — skip all intermediate steps
  - The model learns the full noise pattern at any corruption level
- During generation (inference), the model starts from pure random noise
  - Predicts and removes noise in several steps
  - Each step removes some noise, leaving a cleaner image
  - After enough steps: a coherent image emerges from nothing

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

- Powerful intuition: think of images as points in a space
  - A 1-megapixel image = a point in a million-dimensional space
  - But we'll use a 2D spiral to build intuition
- Real images cluster — they don't fill the whole space
  - Most random pixel combinations look like static, not real photos
  - Real photos occupy a tiny fraction of all possible pixel arrangements
  - In our 2D analogy: real images form a spiral pattern
- Adding noise = random walks (Brownian motion)
  - Each point gets nudged in a random direction
  - Small nudges → points stay near the spiral
  - Big nudges → points scatter everywhere
  - This is exactly what Einstein described with pollen grains in water
- The diffusion model learns to reverse these walks
  - Given a scattered point, which direction leads back to the spiral?
  - Given a noisy image, which direction leads back to a real image?
  - Same question, different dimensionality

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

- The model learns a **vector field** — arrows everywhere in the space
  - Each arrow says: "from this point, go THIS direction to get closer to real data"
  - Collectively, they point scattered noise back toward real images
- This vector field is called the **score function**
  - Mathematical term: the gradient of the log probability of the data
  - Intuitive term: "the direction toward the good stuff"
- Time conditioning — absolutely critical:
  - The model takes in TWO inputs: the noisy data AND a time step t
  - Time step tells it HOW noisy the input is
  - Large t (very noisy): model focuses on big picture — is this a face? A landscape? An animal?
  - Small t (slightly noisy): model focuses on details — sharpen this edge, add texture there
- During generation, time steps go from large to small
  - First passes: establish coarse structure (rough shapes, layout)
  - Middle passes: refine medium-scale features (objects, proportions)
  - Final passes: add fine details (textures, lighting, sharpness)

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

- This is a subtle but crucial point
  - The model learns the AVERAGE direction at each point
  - If many different images could have produced this noisy version, the model points toward their average
  - Following the average direction all the way → you end up at the average image
  - Average of all faces = a blurry, generic face with no distinctive features
- Adding noise during generation breaks this convergence
  - Small random kicks push the generation path off the "average highway"
  - Different random kicks → different final images
  - The noise allows sampling from the full distribution, not just the center
- This is why the same prompt produces different images each time
  - Different random starting noise + different random kicks = different paths
  - Each path ends at a valid, sharp image — just a different one
- The math: this relates to **stochastic differential equations**
  - The noise term ensures proper sampling from the learned distribution
  - Without it, you get the mode/mean instead of diverse samples

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

- DDPM — the original approach — uses random noise at every step
  - Many steps needed (~1000) for good results
  - Random → different image each time, even from same starting noise
  - Slow: 1000 neural network passes per image
- DDIM — a major practical improvement
  - Replaces the random walk with smooth differential equations
  - Follows the learned vector field like a smooth flow
  - No random noise added during generation
  - Deterministic: same starting noise = same final image every time
- Practical benefits:
  - Fewer steps needed — 20 to 50 instead of 1000
  - Huge speedup for generation
  - Same quality and diversity of outputs (diversity comes from different starting noise)
- The math: ordinary differential equations (ODEs) vs. stochastic differential equations (SDEs)
  - Don't need to understand the math — the intuition is smooth flow vs. noisy walk

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

- This is the mechanism that connects your text prompt to the generated image
  - Two runs of the model at each denoising step
  - Run 1: "denoise this, guided by the prompt" → conditioned prediction
  - Run 2: "denoise this, with no prompt at all" → unconditioned prediction
- The difference between the two = what the prompt is contributing
  - Subtract unconditioned from conditioned → isolate the prompt's effect
  - This tells you: "HERE is the direction that represents your text prompt"
- Alpha (guidance scale) amplifies this difference
  - Higher alpha = more influence from the prompt
  - Lower alpha = more "generic" image, less prompt adherence
- The tree example is vivid:
  - As alpha increases, the tree literally grows — more branches, fuller canopy, more tree-like
  - The model is exaggerating the "tree-ness" direction
  - Too much alpha → oversaturated, unrealistic, but unmistakably what you asked for
- This is computationally expensive — 2× the neural network passes
  - But the results are dramatically better than single-pass generation

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

- Negative prompts use the same guidance math, just in the opposite direction
  - Positive prompt: "steer toward this" → amplify
  - Negative prompt: "steer away from this" → subtract
- Practical use cases from WAN 2.1:
  - "extra fingers" — a common artifact in AI-generated images
  - "walking backwards" — a common artifact in AI-generated videos
  - "blurry, low quality, deformed" — general quality boosters
- How it works mathematically:
  - Compute the direction for the negative prompt
  - Subtract it from the generation direction
  - The model actively avoids those features
- This gives users fine-grained control:
  - Positive prompt: what you want
  - Negative prompt: what you don't want
  - Guidance scale: how strongly to enforce both
- Real workflow for AI image generation:
  - Craft your positive prompt
  - Add negative prompts for known problem areas
  - Adjust guidance scale to taste
  - Generate multiple versions with different random seeds

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

- Step back and appreciate how the pieces fit:
  - CLIP: "text and images can live in the same vector space"
  - Diffusion: "you can generate images by reversing noise"
  - Guidance: "you can steer generation using vectors from text"
  - Each piece was developed somewhat independently — they weren't designed as one system
- The unifying thread: high-dimensional geometry
  - Concepts are directions — we saw this with word embeddings ("king - man + woman = queen")
  - CLIP extends this to images — hat direction, style direction, content direction
  - Diffusion uses vector fields — arrows pointing toward real data
  - Guidance manipulates those vectors using text-derived directions
  - Same math, same intuitions, different applications
- "All you need is language" — once concepts become vectors:
  - You can search for images using text (CLIP)
  - You can generate images from text (diffusion + guidance)
  - You can edit images with text (change directions in the space)
  - Language becomes the universal interface to visual content
- Connection to earlier sessions:
  - The embedding directions, dot products, and high-dimensional spaces from the LLM sessions
  - All the same principles at work, extended to a new domain
  - Understanding one deeply gives you intuition for the other

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
