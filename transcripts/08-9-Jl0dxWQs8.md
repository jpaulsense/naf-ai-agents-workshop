# How might LLMs store facts - Deep Learning Chapter 7

**Video ID:** 9-Jl0dxWQs8
**URL:** https://www.youtube.com/watch?v=9-Jl0dxWQs8
**Playlist Position:** 08

---

## Transcript

[00:00] If you feed a large language model the phrase, Michael Jordan plays the sport of blank,
[00:05] and you have it predict what comes next, and it correctly predicts basketball,
[00:09] this would suggest that somewhere, inside its hundreds of billions of parameters,
[00:14] it's baked in knowledge about a specific person and his specific sport.
[00:18] And I think in general, anyone who's played around with one of these
[00:22] models has the clear sense that it's memorized tons and tons of facts.
[00:25] So a reasonable question you could ask is, how exactly does that work?
[00:29] And where do those facts live?
[00:35] Last December, a few researchers from Google DeepMind posted about work on this question,
[00:40] and they were using this specific example of matching athletes to their sports.
[00:44] And although a full mechanistic understanding of how facts are stored remains unsolved,
[00:49] they had some interesting partial results, including the very general high-level
[00:54] conclusion that the facts seem to live inside a specific part of these networks,
[00:58] known fancifully as the multi-layer perceptrons, or MLPs for short.
[01:03] In the last couple of chapters, you and I have been digging into
[01:06] the details behind transformers, the architecture underlying large language models,
[01:10] and also underlying a lot of other modern AI.
[01:13] In the most recent chapter, we were focusing on a piece called Attention.
[01:16] And the next step for you and me is to dig into the details of what happens inside
[01:20] these multi-layer perceptrons, which make up the other big portion of the network.
[01:25] The computation here is actually relatively simple,
[01:28] especially when you compare it to attention.
[01:30] It boils down essentially to a pair of matrix
[01:32] multiplications with a simple something in between.
[01:35] However, interpreting what these computations are doing is exceedingly challenging.
[01:41] Our main goal here is to step through the computations and make them memorable,
[01:45] but I'd like to do it in the context of showing a specific example of how
[01:49] one of these blocks could, at least in principle, store a concrete fact.
[01:53] Specifically, it'll be storing the fact that Michael Jordan plays basketball.
[01:58] I should mention the layout here is inspired by a conversation
[02:00] I had with one of those DeepMind researchers, Neil Nanda.
[02:04] For the most part, I will assume that you've either watched the last two chapters,
[02:08] or otherwise you have a basic sense for what a transformer is,
[02:11] but refreshers never hurt, so here's the quick reminder of the overall flow.
[02:15] You and I have been studying a model that's trained
[02:18] to take in a piece of text and predict what comes next.
[02:21] That input text is first broken into a bunch of tokens,
[02:24] which means little chunks that are typically words or little pieces of words,
[02:29] and each token is associated with a high-dimensional vector,
[02:33] which is to say a long list of numbers.
[02:35] This sequence of vectors then repeatedly passes through two kinds of operation,
[02:40] attention, which allows the vectors to pass information between one another,
[02:44] and then the multilayer perceptrons, the thing that we're gonna dig into today,
[02:49] and also there's a certain normalization step in between.
[02:53] After the sequence of vectors has flowed through many,
[02:56] many different iterations of both of these blocks, by the end,
[03:00] the hope is that each vector has soaked up enough information, both from the context,
[03:04] all of the other words in the input, and also from the general knowledge that
[03:09] was baked into the model weights through training,
[03:12] that it can be used to make a prediction of what token comes next.
[03:16] One of the key ideas that I want you to have in your mind is that all of
[03:20] these vectors live in a very, very high-dimensional space,
[03:23] and when you think about that space, different directions can encode different
[03:27] kinds of meaning.
[03:30] So a very classic example that I like to refer back to is how if you look
[03:34] at the embedding of woman and subtract the embedding of man,
[03:37] and you take that little step and you add it to another masculine noun,
[03:41] something like uncle, you land somewhere very,
[03:43] very close to the corresponding feminine noun.
[03:46] In this sense, this particular direction encodes gender information.
[03:51] The idea is that many other distinct directions in this super high-dimensional
[03:55] space could correspond to other features that the model might want to represent.
[04:01] In a transformer, these vectors don't merely encode the meaning of a single word, though.
[04:06] As they flow through the network, they imbibe a much richer meaning based
[04:10] on all the context around them, and also based on the model's knowledge.
[04:15] Ultimately, each one needs to encode something far,
[04:18] far beyond the meaning of a single word, since it needs to be sufficient to
[04:22] predict what will come next.
[04:24] We've already seen how attention blocks let you incorporate context,
[04:28] but a majority of the model parameters actually live inside the MLP blocks,
[04:32] and one thought for what they might be doing is that they offer extra capacity
[04:37] to store facts.
[04:38] Like I said, the lesson here is gonna center on the concrete toy example
[04:42] of how exactly it could store the fact that Michael Jordan plays basketball.
[04:47] Now, this toy example is gonna require that you and I make
[04:49] a couple of assumptions about that high-dimensional space.
[04:52] First, we'll suppose that one of the directions represents the idea of a first name
[04:56] Michael, and then another nearly perpendicular direction represents the idea of the
[05:01] last name Jordan, and then yet a third direction will represent the idea of basketball.
[05:07] So specifically, what I mean by this is if you look in the network and
[05:11] you pluck out one of the vectors being processed,
[05:13] if its dot product with this first name Michael direction is one,
[05:17] that's what it would mean for the vector to be encoding the idea of a
[05:20] person with that first name.
[05:23] Otherwise, that dot product would be zero or negative,
[05:26] meaning the vector doesn't really align with that direction.
[05:29] And for simplicity, let's completely ignore the very reasonable
[05:32] question of what it might mean if that dot product was bigger than one.
[05:36] Similarly, its dot product with these other directions would
[05:39] tell you whether it represents the last name Jordan or basketball.
[05:44] So let's say a vector is meant to represent the full name, Michael Jordan,
[05:48] then its dot product with both of these directions would have to be one.
[05:53] Since the text Michael Jordan spans two different tokens,
[05:56] this would also mean we have to assume that an earlier attention block has successfully
[06:01] passed information to the second of these two vectors so as to ensure that it can
[06:05] encode both names.
[06:07] With all of those as the assumptions, let's now dive into the meat of the lesson.
[06:11] What happens inside a multilayer perceptron?
[06:17] You might think of this sequence of vectors flowing into the block, and remember,
[06:21] each vector was originally associated with one of the tokens from the input text.
[06:26] What's gonna happen is that each individual vector from that sequence
[06:29] goes through a short series of operations, we'll unpack them in just a moment,
[06:33] and at the end, we'll get another vector with the same dimension.
[06:36] That other vector is gonna get added to the original one that flowed in,
[06:40] and that sum is the result flowing out.
[06:43] This sequence of operations is something you apply to every vector in the sequence,
[06:47] associated with every token in the input, and it all happens in parallel.
[06:52] In particular, the vectors don't talk to each other in this step,
[06:54] they're all kind of doing their own thing.
[06:56] And for you and me, that actually makes it a lot simpler,
[06:59] because it means if we understand what happens to just one of the
[07:02] vectors through this block, we effectively understand what happens to all of them.
[07:07] When I say this block is gonna encode the fact that Michael Jordan plays basketball,
[07:11] what I mean is that if a vector flows in that encodes first name Michael and last
[07:15] name Jordan, then this sequence of computations will produce something that includes
[07:19] that direction basketball, which is what will add on to the vector in that position.
[07:25] The first step of this process looks like multiplying that vector by a very big matrix.
[07:30] No surprises there, this is deep learning.
[07:32] And this matrix, like all of the other ones we've seen,
[07:35] is filled with model parameters that are learned from data,
[07:37] which you might think of as a bunch of knobs and dials that get tweaked and
[07:41] tuned to determine what the model behavior is.
[07:44] Now, one nice way to think about matrix multiplication is to imagine each row of
[07:48] that matrix as being its own vector, and taking a bunch of dot products between
[07:52] those rows and the vector being processed, which I'll label as E for embedding.
[07:57] For example, suppose that very first row happened to equal
[08:00] this first name Michael direction that we're presuming exists.
[08:04] That would mean that the first component in this output, this dot product right here,
[08:09] would be one if that vector encodes the first name Michael,
[08:12] and zero or negative otherwise.
[08:15] Even more fun, take a moment to think about what it would mean if that
[08:19] first row was this first name Michael plus last name Jordan direction.
[08:23] And for simplicity, let me go ahead and write that down as M plus J.
[08:28] Then, taking a dot product with this embedding E,
[08:30] things distribute really nicely, so it looks like M dot E plus J dot E.
[08:34] And notice how that means the ultimate value would be two if the vector encodes the
[08:39] full name Michael Jordan, and otherwise it would be one or something smaller than one.
[08:45] And that's just one row in this matrix.
[08:47] You might think of all of the other rows as in parallel asking some other kinds of
[08:51] questions, probing at some other sorts of features of the vector being processed.
[08:56] Very often this step also involves adding another vector to the output,
[08:59] which is full of model parameters learned from data.
[09:02] This other vector is known as the bias.
[09:05] For our example, I want you to imagine that the value of this
[09:08] bias in that very first component is negative one,
[09:11] meaning our final output looks like that relevant dot product, but minus one.
[09:16] You might very reasonably ask why I would want you to assume that the
[09:19] model has learned this, and in a moment you'll see why it's very clean
[09:23] and nice if we have a value here which is positive if and only if a vector
[09:28] encodes the full name Michael Jordan, and otherwise it's zero or negative.
[09:33] The total number of rows in this matrix, which is something
[09:36] like the number of questions being asked, in the case of GPT-3,
[09:39] whose numbers we've been following, is just under 50,000.
[09:43] In fact, it's exactly four times the number of dimensions in this embedding space.
[09:46] That's a design choice.
[09:47] You could make it more, you could make it less,
[09:49] but having a clean multiple tends to be friendly for hardware.
[09:52] Since this matrix full of weights maps us into a higher dimensional space,
[09:56] I'm gonna give it the shorthand W up.
[09:59] I'll continue labeling the vector we're processing as E,
[10:02] and let's label this bias vector as B up and put that all back down in the diagram.
[10:09] At this point, a problem is that this operation is purely linear,
[10:12] but language is a very non-linear process.
[10:15] If the entry that we're measuring is high for Michael plus Jordan,
[10:19] it would also necessarily be somewhat triggered by Michael plus Phelps
[10:23] and also Alexis plus Jordan, despite those being unrelated conceptually.
[10:28] What you really want is a simple yes or no for the full name.
[10:32] So the next step is to pass this large intermediate
[10:35] vector through a very simple non-linear function.
[10:38] A common choice is one that takes all of the negative values and
[10:41] maps them to zero and leaves all of the positive values unchanged.
[10:46] And continuing with the deep learning tradition of overly fancy names,
[10:50] this very simple function is often called the rectified linear unit, or ReLU for short.
[10:56] Here's what the graph looks like.
[10:58] So taking our imagined example where this first entry of the intermediate vector is one,
[11:03] if and only if the full name is Michael Jordan and zero or negative otherwise,
[11:07] after you pass it through the ReLU, you end up with a very clean value where
[11:12] all of the zero and negative values just get clipped to zero.
[11:16] So this output would be one for the full name Michael Jordan and zero otherwise.
[11:20] In other words, it very directly mimics the behavior of an AND gate.
[11:25] Often models will use a slightly modified function that's called the GELU,
[11:29] which has the same basic shape, it's just a bit smoother.
[11:32] But for our purposes, it's a little bit cleaner if we only think about the ReLU.
[11:36] Also, when you hear people refer to the neurons of a transformer,
[11:40] they're talking about these values right here.
[11:42] Whenever you see that common neural network picture with a layer of dots and a
[11:47] bunch of lines connecting to the previous layer, which we had earlier in this series,
[11:52] that's typically meant to convey this combination of a linear step,
[11:56] a matrix multiplication, followed by some simple term-wise nonlinear function like a ReLU.
[12:02] You would say that this neuron is active whenever this value
[12:05] is positive and that it's inactive if that value is zero.
[12:10] The next step looks very similar to the first one.
[12:12] You multiply by a very large matrix and you add on a certain bias term.
[12:16] In this case, the number of dimensions in the output is back down to the size of
[12:21] that embedding space, so I'm gonna go ahead and call this the down projection matrix.
[12:26] And this time, instead of thinking of things row by row,
[12:28] it's actually nicer to think of it column by column.
[12:31] You see, another way that you can hold matrix multiplication in your head is to
[12:36] imagine taking each column of the matrix and multiplying it by the corresponding
[12:40] term in the vector that it's processing and adding together all of those rescaled columns.
[12:46] The reason it's nicer to think about this way is because here the columns have the same
[12:51] dimension as the embedding space, so we can think of them as directions in that space.
[12:56] For instance, we will imagine that the model has learned to make that
[12:59] first column into this basketball direction that we suppose exists.
[13:04] What that would mean is that when the relevant neuron in that first position is active,
[13:08] we'll be adding this column to the final result.
[13:11] But if that neuron was inactive, if that number was zero, then this would have no effect.
[13:16] And it doesn't just have to be basketball.
[13:18] The model could also bake into this column and many other features that
[13:21] it wants to associate with something that has the full name Michael Jordan.
[13:26] And at the same time, all of the other columns in this matrix are telling you
[13:31] what will be added to the final result if the corresponding neuron is active.
[13:37] And if you have a bias in this case, it's something that you're
[13:40] just adding every single time, regardless of the neuron values.
[13:44] You might wonder what's that doing.
[13:45] As with all parameter-filled objects here, it's kind of hard to say exactly.
[13:49] Maybe there's some bookkeeping that the network needs to do,
[13:52] but you can feel free to ignore it for now.
[13:54] Making our notation a little more compact again,
[13:57] I'll call this big matrix W down and similarly call that bias vector B down and
[14:02] put that back into our diagram.
[14:04] Like I previewed earlier, what you do with this final result is add it to the vector
[14:09] that flowed into the block at that position and that gets you this final result.
[14:13] So for example, if the vector flowing in encoded both first name Michael and last name
[14:19] Jordan, then because this sequence of operations will trigger that AND gate,
[14:23] it will add on the basketball direction, so what pops out will encode all of those
[14:28] together.
[14:29] And remember, this is a process happening to every one of those vectors in parallel.
[14:34] In particular, taking the GPT-3 numbers, it means that this block doesn't just
[14:39] have 50,000 neurons in it, it has 50,000 times the number of tokens in the input.
[14:48] So that is the entire operation, two matrix products,
[14:51] each with a bias added and a simple clipping function in between.
[14:56] Any of you who watched the earlier videos of the series will recognize this
[14:59] structure as the most basic kind of neural network that we studied there.
[15:03] In that example, it was trained to recognize handwritten digits.
[15:06] Over here, in the context of a transformer for a large language model,
[15:10] this is one piece in a larger architecture and any attempt to interpret
[15:15] what exactly it's doing is heavily intertwined with the idea of encoding
[15:19] information into vectors of a high-dimensional embedding space.
[15:24] That is the core lesson, but I do wanna step back and reflect on two different things,
[15:28] the first of which is a kind of bookkeeping, and the second of which
[15:32] involves a very thought-provoking fact about higher dimensions that
[15:35] I actually didn't know until I dug into transformers.
[15:41] In the last two chapters, you and I started counting up the total number of parameters
[15:45] in GPT-3 and seeing exactly where they live, so let's quickly finish up the game here.
[15:51] I already mentioned how this up projection matrix has just under 50,000 rows and
[15:56] that each row matches the size of the embedding space, which for GPT-3 is 12,288.
[16:03] Multiplying those together, it gives us 604 million parameters just for that matrix,
[16:08] and the down projection has the same number of parameters just with a transposed shape.
[16:14] So together, they give about 1.2 billion parameters.
[16:18] The bias vector also accounts for a couple more parameters,
[16:20] but it's a trivial proportion of the total, so I'm not even gonna show it.
[16:24] In GPT-3, this sequence of embedding vectors flows through not one,
[16:29] but 96 distinct MLPs, so the total number of parameters devoted
[16:34] to all of these blocks adds up to about 116 billion.
[16:38] This is around 2 thirds of the total parameters in the network,
[16:42] and when you add it to everything that we had before, for the attention blocks,
[16:46] the embedding, and the unembedding, you do indeed get that grand total of 175
[16:50] billion as advertised.
[16:53] It's probably worth mentioning there's another set of parameters associated
[16:56] with those normalization steps that this explanation has skipped over,
[16:59] but like the bias vector, they account for a very trivial proportion of the total.
[17:05] As to that second point of reflection, you might be wondering if
[17:09] this central toy example we've been spending so much time on
[17:12] reflects how facts are actually stored in real large language models.
[17:16] It is true that the rows of that first matrix can be thought of as
[17:19] directions in this embedding space, and that means the activation of each
[17:23] neuron tells you how much a given vector aligns with some specific direction.
[17:27] It's also true that the columns of that second matrix tell
[17:30] you what will be added to the result if that neuron is active.
[17:34] Both of those are just mathematical facts.
[17:37] However, the evidence does suggest that individual neurons very rarely
[17:41] represent a single clean feature like Michael Jordan,
[17:44] and there may actually be a very good reason this is the case,
[17:48] related to an idea floating around interpretability researchers these
[17:52] days known as superposition.
[17:54] This is a hypothesis that might help to explain both why the models are
[17:58] especially hard to interpret and also why they scale surprisingly well.
[18:03] The basic idea is that if you have an n-dimensional space and you wanna
[18:07] represent a bunch of different features using directions that are all
[18:11] perpendicular to one another in that space, you know,
[18:14] that way if you add a component in one direction,
[18:16] it doesn't influence any of the other directions,
[18:19] then the maximum number of vectors you can fit is only n, the number of dimensions.
[18:24] To a mathematician, actually, this is the definition of dimension.
[18:28] But where it gets interesting is if you relax that
[18:30] constraint a little bit and you tolerate some noise.
[18:34] Say you allow those features to be represented by vectors that aren't exactly
[18:38] perpendicular, they're just nearly perpendicular, maybe between 89 and 91 degrees apart.
[18:44] If we were in two or three dimensions, this makes no difference.
[18:48] That gives you hardly any extra wiggle room to fit more vectors in,
[18:51] which makes it all the more counterintuitive that for higher dimensions,
[18:55] the answer changes dramatically.
[18:57] I can give you a really quick and dirty illustration of this using some
[19:01] scrappy Python that's going to create a list of 100-dimensional vectors,
[19:06] each one initialized randomly, and this list is going to contain 10,000 distinct vectors,
[19:11] so 100 times as many vectors as there are dimensions.
[19:15] This plot right here shows the distribution of angles between pairs of these vectors.
[19:20] So because they started at random, those angles could be anything from 0 to 180 degrees,
[19:25] but you'll notice that already, even just for random vectors,
[19:28] there's this heavy bias for things to be closer to 90 degrees.
[19:32] Then what I'm going to do is run a certain optimization process that iteratively nudges
[19:37] all of these vectors so that they try to become more perpendicular to one another.
[19:42] After repeating this many different times, here's
[19:44] what the distribution of angles looks like.
[19:47] We have to actually zoom in on it here because all of the possible angles
[19:51] between pairs of vectors sit inside this narrow range between 89 and 91 degrees.
[19:58] In general, a consequence of something known as the Johnson-Lindenstrauss
[20:02] lemma is that the number of vectors you can cram into a space that are nearly
[20:06] perpendicular like this grows exponentially with the number of dimensions.
[20:11] This is very significant for large language models,
[20:14] which might benefit from associating independent ideas with nearly
[20:18] perpendicular directions.
[20:20] It means that it's possible for it to store many,
[20:22] many more ideas than there are dimensions in the space that it's allotted.
[20:27] This might partially explain why model performance seems to scale so well with size.
[20:32] A space that has 10 times as many dimensions can store way,
[20:36] way more than 10 times as many independent ideas.
[20:40] And this is relevant not just to that embedding space where the vectors
[20:43] flowing through the model live, but also to that vector full of neurons
[20:47] in the middle of that multilayer perceptron that we just studied.
[20:50] That is to say, at the sizes of GPT-3, it might not just be probing at 50,000 features,
[20:56] but if it instead leveraged this enormous added capacity by using
[20:59] nearly perpendicular directions of the space, it could be probing at many,
[21:04] many more features of the vector being processed.
[21:07] But if it was doing that, what it means is that individual
[21:10] features aren't gonna be visible as a single neuron lighting up.
[21:14] It would have to look like some specific combination of neurons instead, a superposition.
[21:20] For any of you curious to learn more, a key relevant search term here is sparse
[21:24] autoencoder, which is a tool that some of the interpretability people use to try to
[21:28] extract what the true features are, even if they're very superimposed on all these
[21:32] neurons.
[21:33] I'll link to a couple really great anthropic posts all about this.
[21:37] At this point, we haven't touched every detail of a transformer,
[21:40] but you and I have hit the most important points.
[21:43] The main thing that I wanna cover in a next chapter is the training process.
[21:48] On the one hand, the short answer for how training works is that it's all
[21:51] backpropagation, and we covered backpropagation in a separate context with earlier
[21:55] chapters in the series.
[21:57] But there is more to discuss, like the specific cost function used for language models,
[22:02] the idea of fine-tuning using reinforcement learning with human feedback,
[22:06] and the notion of scaling laws.
[22:08] Quick note for the active followers among you,
[22:11] there are a number of non-machine learning-related videos that I'm excited to
[22:14] sink my teeth into before I make that next chapter, so it might be a while,
[22:18] but I do promise it'll come in due time.
[22:35] Thank you.
