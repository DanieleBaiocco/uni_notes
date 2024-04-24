## Bidirectional transformer
Bert is a bidirectional transformer encoder model, in which self-attention is not only performed from left to right, but also considering the other words.
How to train these types of models?

Causal transformer language models are trained by making them iteratively predict the next word in a text.  But eliminating the causal mask makes the guess-the-next-word language modeling task trivial since the answer is now directly available from the context.\*So*?
Instead of trying to predict the next word, the model learns to perform a **fill-in-the-blank** task, technically called the cloze tas.
During training the model is deprived of one or more elements of an input sequence and must generate a probability distribution over the vocabulary for each of the missing items. 
We then use the **cross-entropy** loss from each of the model’s predictions to drive the learning process.

### Masking words
The original approach to training bidirectional encoders is called **Masked Language
Modeling**.

Here, the MLM model is presented with a series of sentences from the training corpus where a random sample of tokens from each training sequence is selected for use in the learning task.
Once chosen, a token is used in ONE of 3 ways:
1. It is replaced with the unique vocabulary token \[MASK\]
2. It is replaced with another token from the vocabulary, randomly sampled based on token unigram probabilities
3. t is left unchanged

In BERT, 15% of the input tokens in a training sequence are sampled for learning.
Of these
1. 80% are replaced with \[MASK]
2. 10% are replaced with randomly selected tokens 
3. the remaining 10% are left unchanged
The MLM training objective is to predict the original inputs for each of the
masked tokens using a bidirectional encoder of the kind described in the last section

Note that all of the input tokens play a role in the self-attention process, but only the sampled tokens are used for learning. This means that when computing the loss, it is only computed on the outputs bounded to these sampled tokens.

More specifically:
1. The original input sequence is first tokenized using a subword model
2. The sampled items which drive the learning process are chosen from among
the set of tokenized inputs
3. Word embeddings for all of the tokens in the input are retrieved from the word embedding matrix and then combined with positional embeddings to form the input to the transformer
![[masked_training.png]]
### Masking spans
For many NLP applications, the natural unit of interest may be larger than a single
word (or token). 
A span is a contiguous sequence of one or more words selected from a train-
ing text, prior to subword tokenization. In span-based masking, a set of randomly
selected spans from a training sequence are chosen.
In the SpanBERT work that originated this technique
1. a span length is first chosen by sampling from a geometric distribution that is biased towards shorter spans and with an upper bound of 10
2. Given this span length, a starting location consistent with the desired span length and the length of the input is sampled uniformly
3. Once a span is chosen for masking, all the words within the span are substituted according to the same regime used in BERT: 80% of the time the span elements are substituted with the \[MASK\] token, 10% of the time they are replaced by randomly sampled words from the vocabulary, and 10% of the time they are left as is. Note that this substitution process is done at the span level—all the tokens in a given span are substituted using the same method As with BERT, the total token substitution is limited to 15% of the training sequence input
4. Having selected and masked the training span, the input is passed through the standard transformer architecture to generate contextualized representations of the input tokens
The SpanBERT learning objective augments the MLM
objective with a boundary oriented component called the Span Boundary Objective
(SBO). The SBO relies on a model’s ability to predict the words within a masked
span from the words immediately preceding and following it. 
SpanBERT loss is computed in this way:
For a single token x we have 
L(x) = L<sub>MLM</sub> (x) + L<sub>SBO</sub>(x) 
where  L<sub>SBO</sub>(x) = −logP(x|x<sub>s</sub>, x<sub>e</sub>, p<sub>x</sub>) in which s denotes the position of the word before the span and e denotes the word after the end, and p<sub>x</sub> is the positional embedding of token x

This L<sub>SBO</sub> prediction for a given position i within the span is produced
by concatenating the output embeddings for words s and e span boundary vectors
with a positional embedding for position i and passing the result through a 2-layer
feedforward network
s = FFN(\[y<sub>s</sub>; y<sub>e</sub>; p<sub>i−s+1</sub>]
z = softmax(Es)
Look at this image: 
![[span_mask_loss.png]]

### Next sentence prediction
The focus of masked-based learning is on predicting words from surrounding con-
texts with the goal of producing effective word-level representations
But there are tasks like paraphrase detection (detecting if two sen-
tences have similar meanings), entailment (detecting if the meanings of two sen-
tences entail or contradict each other) or discourse coherence (deciding if two neigh-
boring sentences form a coherent discourse).
To capture the kind of knowledge required for applications such as these, BERT
introduced a second learning objective called Next Sentence Prediction (NSP).
In this task, the model is presented with pairs of sentences and is asked to predict
whether each pair consists of an actual pair of adjacent sentences from the training
corpus or a pair of unrelated sentences.
In BERT, 50% of the training pairs consisted
of positive pairs, and in the other 50% the second sentence of a pair was randomly
selected from elsewhere in the corpus. The NSP loss is based on how well the model
can distinguish true pairs from random pairs.

To facilitate NSP training, BERT introduces two new tokens to the input repre-
sentation (tokens that will prove useful for fine-tuning as well). After tokenizing the
input with the subword model, the token \[CLS] is prepended to the input sentence
pair, and the token \[SEP] is placed between the sentences and after the final token of
the second sentence. Finally, embeddings representing the first and second segments
of the input are added to the word and positional embeddings to allow the model to
more easily distinguish the input sentences.

During training, the output vector from the final layer associated with the /[CLS]
token  (that is the first token, remembder) represents the next sentence prediction (so it tells whether they are actual pairs of adjacent sentences or not).  As with the MLM objective, a learned
set of classification weights W<sub>NSP</sub> ∈ R <sup>2×d<sub>h</sub></sup> is used to produce a two-class prediction
from the raw \[CLS] vector (it gives as output a y<sub>i</sub> that contains two values aka a proba dist)
y<sub>i</sub> = softmax(W<sub>NSP</sub> h<sub>i</sub>) with i index i think of the training set, and h hidden state produced from token \[CLS].
![[next_sentence_prediction_training.png]]


To train the original BERT models, pairs of sentences were selected from the
training corpus according to the next sentence prediction 50/50 scheme. Pairs were
sampled so that their combined length was less than the 512 token input. Tokens
within these sentence pairs were then masked using the MLM approach with the
combined loss from the MLM and NSP objectives used for a final loss. Approx-
imately 40 passes (epochs) over the training data was required for the model to
converge.

The result of this pretraining process consists of both **learned word embeddings**,
as well as all the **parameters of the bidirectional encoder** that are used to **produce
contextual embeddings for novel inputs** (because each time we insert a new phrase in this model, for each token it generates a contextual word embedding that is the hidden state associated with it)

## Contextual embeddings
Given a pretrained language model and a novel input sentence, we can think of the
output of the model as constituting contextual embeddings for each token in the input. These contextual embeddings can be used as a contextual representation of
the meaning of the input token for any task requiring the meaning of word.
ontextual embeddings are thus vectors representing some aspect of the meaning
of a token in context. For example, given a sequence of input tokens x1, ..., xn, we can
use the output vector yi from the final layer of the model as a representation of the
meaning of token xi in the context of sentence x1, ..., xn. Or instead of just using the
vector yi from the final layer of the model, it’s common to compute a representation
for xi by averaging the output tokens yi from each of the last four layers of the model.
Just as we used static embeddings like word2vec to represent the meaning of
words, we can use contextual embeddings as representations of word meanings in
context for any task that might require a model of word meaning. 

Where static
embeddings represent the meaning of word types (vocabulary entries), contextual
embeddings represent the meaning of word tokens: instances of a particular word
type in a particular context. Contextual embeddings can thus be used for tasks like
measuring the semantic similarity of two words in context, and are useful in linguis-
tic tasks that require models of word meaning.


### Transfer Learning through Fine-Tuning
ine-tuning facilitates the creation of applications on top of pre-trained models through the addition of a small set of application-specific parameters.The fine-tuning process consists of using labeled data from the application to train these additional application-specific parameter. Typically, this training will either freeze or make only minimal adjustments to the pretrained language model parameters.

#### Sequence Classification
An additional vector is added to the model to stand for the entire
sequence. This vector is sometimes called the sentence embedding since it refers to the entire sequence.
In BERT, the \[CLS] token plays the role of this embedding. This unique token
is added to the vocabulary and is prepended to the start of all input sequences, both
during pretraining and encoding. The output vector in the final layer of the model
for the \[CLS] input represents the entire input sequence. It serves as the input to
a classifier head, a logistic regression or neural network classifier that makes the decision.

A simple approach to fine-tuning a classifier for this application involves learning a set of
weights, W<sub>C</sub>, to map the output vector for the \[CLS] token, y<sub>CLS</sub> to a set of scores
over the possible sentiment classes.  Assuming a three-way sentiment classification
task (positive, negative, neutral) and dimensionality dh for the size of the language
model hidden layers gives W<sub>C</sub> ∈ R<sup>3×d<sub>h</sub></sup>.
lassification of unseen documents proceeds by passing the input text through the pretrained language model to generate y<sub>CLS</sub>, multiplying it by W<sub>C</sub>, and finally passing the resulting vector through a softmax.
y = softmax(W<sub>C</sub>y<sub>CLS</sub>)

Finetuning the values in W<sub>C</sub> requires supervised training data consisting of input
sequences labeled with the appropriate class. Training proceeds in the usual way;
cross-entropy loss between the softmax output and the correct answer is used to
drive the learning that produces W<sub>C</sub>

A key difference from what we’ve seen earlier with neural classifiers is that this
loss can be used to not only learn the weights of the classifier, but also to update the
weights for the pretrained language model itself. 

In practice, reasonable classification performance is typically achieved with only minimal changes to the language model parameters, often limited to updates over the final few layers of the trans-
former.
![[sentence_classification_fine_tuning.png]]
#### Pair wise sequence classification
It is useful for my project (second assignment)
To fine-tune a classifier for the MultiNLI task, we pass the premise/hypothesis
pairs through a bidirectional encoder as described above and use the output vector
for the \[CLS] token as the input to the classification head. As with ordinary sequence
classification, this head provides the input to a three-way classifier that can be trained
on the MultiNLI training corpus.

#### Sequence Labelling (es. Pos tagging)
Here, the final output vector corresponding to each input token is passed to a classifier that produces a softmax distribution over the possible set of tags. 
Again, assuming a simple classifier
consisting of a single feedforward layer followed by a softmax, the set of weights
to be learned for this additional layer is W<sub>K</sub> ∈ R<sup>k×d<sub>h</sub></sup>, where k is the number of
possible tags for the task. As with RNNs, a greedy approach, where the argmax tag
for each token is taken as a likely answer, can be used to generate the final output
tag sequence.

y<sub>i</sub> = softmax(W<sub>K</sub>z<sub>i</sub>) 
t<sub>i </sub>= argmax<sub>k</sub>(y<sub>i</sub>)

#### Fine-tuning for Span-Based Applications
In span-oriented applications the focus is on generating
and operating with representations of contiguous sequences of tokens. Typical op-
erations include identifying spans of interest, classifying spans according to some
labeling scheme, and determining relations among discovered spans.

C'e' molto da dire ma non ne ho voglia. Guarda libro (sembra un concetto poco importante)
![[span_based_fine_tuning.png]]

FINE CAPITOLO


