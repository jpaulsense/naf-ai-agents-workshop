# Large Language Models explained briefly

**Video ID:** LPZh9BOjkQs
**URL:** https://www.youtube.com/watch?v=LPZh9BOjkQs
**Playlist Position:** 05

---

## Transcript

[00:01] Imagine you happen across a short movie script that
[00:03] describes a scene between a person and their AI assistant.
[00:07] The script has what the person asks the AI, but the AI's response has been torn off.
[00:13] Suppose you also have this powerful magical machine that can take
[00:16] any text and provide a sensible prediction of what word comes next.
[00:21] You could then finish the script by feeding in what you have to the machine,
[00:25] seeing what it would predict to start the AI's answer,
[00:28] and then repeating this over and over with a growing script completing the dialogue.
[00:33] When you interact with a chatbot, this is exactly what's happening.
[00:37] A large language model is a sophisticated mathematical function
[00:40] that predicts what word comes next for any piece of text.
[00:44] Instead of predicting one word with certainty, though,
[00:47] what it does is assign a probability to all possible next words.
[00:51] To build a chatbot, you lay out some text that describes an interaction between a user
[00:56] and a hypothetical AI assistant, add on whatever the user types in as the first part of
[01:02] the interaction, and then have the model repeatedly predict the next word that such a
[01:07] hypothetical AI assistant would say in response, and that's what's presented to the user.
[01:13] In doing this, the output tends to look a lot more natural if
[01:16] you allow it to select less likely words along the way at random.
[01:20] So what this means is even though the model itself is deterministic,
[01:23] a given prompt typically gives a different answer each time it's run.
[01:28] Models learn how to make these predictions by processing an enormous amount of text,
[01:32] typically pulled from the internet.
[01:34] For a standard human to read the amount of text that was used to train GPT-3,
[01:39] for example, if they read non-stop 24-7, it would take over 2600 years.
[01:44] Larger models since then train on much, much more.
[01:48] You can think of training a little bit like tuning the dials on a big machine.
[01:52] The way that a language model behaves is entirely determined by these
[01:56] many different continuous values, usually called parameters or weights.
[02:01] Changing those parameters will change the probabilities
[02:04] that the model gives for the next word on a given input.
[02:07] What puts the large in large language model is how
[02:10] they can have hundreds of billions of these parameters.
[02:15] No human ever deliberately sets those parameters.
[02:18] Instead, they begin at random, meaning the model just outputs gibberish,
[02:22] but they're repeatedly refined based on many example pieces of text.
[02:27] One of these training examples could be just a handful of words,
[02:30] or it could be thousands, but in either case, the way this works is to
[02:34] pass in all but the last word from that example into the model and
[02:38] compare the prediction that it makes with the true last word from the example.
[02:43] An algorithm called backpropagation is used to tweak all of the parameters
[02:47] in such a way that it makes the model a little more likely to choose
[02:51] the true last word and a little less likely to choose all the others.
[02:55] When you do this for many, many trillions of examples,
[02:58] not only does the model start to give more accurate predictions on the training data,
[03:03] but it also starts to make more reasonable predictions on text that it's never
[03:07] seen before.
[03:09] Given the huge number of parameters and the enormous amount of training data,
[03:13] the scale of computation involved in training a large language model is mind-boggling.
[03:19] To illustrate, imagine that you could perform one
[03:22] billion additions and multiplications every single second.
[03:26] How long do you think it would take for you to do all of the
[03:29] operations involved in training the largest language models?
[03:33] Do you think it would take a year?
[03:36] Maybe something like 10,000 years?
[03:39] The answer is actually much more than that.
[03:41] It's well over 100 million years.
[03:45] This is only part of the story, though.
[03:47] This whole process is called pre-training.
[03:49] The goal of auto-completing a random passage of text from the
[03:52] internet is very different from the goal of being a good AI assistant.
[03:56] To address this, chatbots undergo another type of training,
[04:00] just as important, called reinforcement learning with human feedback.
[04:04] Workers flag unhelpful or problematic predictions,
[04:07] and their corrections further change the model's parameters,
[04:11] making them more likely to give predictions that users prefer.
[04:14] Looking back at the pre-training, though, this staggering amount of
[04:18] computation is only made possible by using special computer chips that
[04:23] are optimized for running many operations in parallel, known as GPUs.
[04:28] However, not all language models can be easily parallelized.
[04:32] Prior to 2017, most language models would process text one word at a time,
[04:36] but then a team of researchers at Google introduced a new model known as the transformer.
[04:43] Transformers don't read text from the start to the finish,
[04:46] they soak it all in at once, in parallel.
[04:49] The very first step inside a transformer, and most other language models for that matter,
[04:54] is to associate each word with a long list of numbers.
[04:57] The reason for this is that the training process only works with continuous values,
[05:02] so you have to somehow encode language using numbers,
[05:05] and each of these lists of numbers may somehow encode the meaning of the
[05:09] corresponding word.
[05:10] What makes transformers unique is their reliance
[05:13] on a special operation known as attention.
[05:16] This operation gives all of these lists of numbers a chance to talk to one another
[05:21] and refine the meanings they encode based on the context around, all done in parallel.
[05:27] For example, the numbers encoding the word bank might be changed based on the
[05:31] context surrounding it to somehow encode the more specific notion of a riverbank.
[05:37] Transformers typically also include a second type of operation known
[05:41] as a feed-forward neural network, and this gives the model extra
[05:44] capacity to store more patterns about language learned during training.
[05:49] All of this data repeatedly flows through many different iterations of
[05:53] these two fundamental operations, and as it does so,
[05:56] the hope is that each list of numbers is enriched to encode whatever
[06:00] information might be needed to make an accurate prediction of what word
[06:04] follows in the passage.
[06:07] At the end, one final function is performed on the last vector in this sequence,
[06:11] which now has had a chance to be influenced by all the other context from the input text,
[06:16] as well as everything the model learned during training,
[06:19] to produce a prediction of the next word.
[06:22] Again, the model's prediction looks like a probability for every possible next word.
[06:28] Although researchers design the framework for how each of these steps work,
[06:32] it's important to understand that the specific behavior is an emergent phenomenon
[06:37] based on how those hundreds of billions of parameters are tuned during training.
[06:42] This makes it incredibly challenging to determine
[06:45] why the model makes the exact predictions that it does.
[06:48] What you can see is that when you use large language model predictions to autocomplete
[06:53] a prompt, the words that it generates are uncannily fluent, fascinating, and even useful.
[07:05] If you're a new viewer and you're curious about more details on how
[07:08] transformers and attention work, boy do I have some material for you.
[07:12] One option is to jump into a series I made about deep learning,
[07:16] where we visualize and motivate the details of attention and all the other steps
[07:20] in a transformer.
[07:22] Also, on my second channel I just posted a talk I gave a couple
[07:25] months ago about this topic for the company TNG in Munich.
[07:29] Sometimes I actually prefer the content I make as a casual talk rather than a produced
[07:33] video, but I leave it up to you which one of these feels like the better follow-on.
