## IR-based QA
The goal of IR-based QA (sometimes called open domain QA) is to answer a user’s question by finding short text segments from the web or some other large collection of documents.
The dominant paradigm for IR-based QA is the **retrieve** and **read** model

In the first stage of this 2-stage model we retrieve relevant passages from a text collection, usually using a search engines of the type we saw in the previous section. In the second stage, a neural reading comprehension algorithm passes over each passage and finds spans that are likely to answer the question
![[IRQA.png]]
### IR-based QA: Reader (Answer Span Extraction)
The answer extraction task is commonly modeled by **span labeling**: identifying in the passage a span (a continuous string of text) **that constitutes an answer**.

If each span a starts at position a<sub>s</sub> and ends at position a<sub>e</sub>, we make the simplifying assumption that this probability can be estimated as P(a|q, p) = P<sub>start</sub>(a<sub>s</sub> |q, p)P<sub>end</sub>(a<sub>e</sub>|q, p). Thus for for each token p<sub>i</sub> in the passage we’ll compute two probabilities: p<sub>start</sub>(i) that p<sub>i</sub> is the start of the answer span, and p<sub>end</sub>(i) that p<sub>i</sub> is the end of the answer span.

A standard baseline algorithm for reading comprehension is to pass the question and passage to any encoder like BERT, as strings separated with a \[SEP\] token, resulting in an encoding token embedding for every passage token p<sub>i</sub>.

![[extractiveQA.png]]

For span-based question answering, we represent the question as the first sequence and the passage as the second sequence. We’ll also need to add a linear layer that will be trained in the fine-tuning phase to predict the start and end position of the span.
The score of a candidate span from position i to j is S · pi +E · pj , and the highest scoring span in which j ≥ i is chosen is the model prediction.

The training loss for fine-tuning is the negative sum of the log-likelihoods of the correct start and end positions for each instance:

L = −log P<sub>start i</sub> −log P<sub>end i</sub>

Many datasets (like SQuAD 2.0 and Natural Questions) also contain (question, passage) pairs in which the answer is not contained in the passage. We thus also need a way to estimate the probability that the answer to a question is not in the document. This is standardly done by treating questions with no answer as having the [CLS] token as the answer, and hence the answer span start and end index will point at [CLS]

## Entity Linking
Quello visto finora era il **primo** paradigma. Ora andiamo al secondo. E' knowledge based. quelli knowledge based si basano su ENTITY LINKING che e' quello che spieghero adesso.

**Entity linking** is the task of associating a mention in text with the representation of some real-world entity in an ontology. Quindi associare una mention a una categoria/entita del mondo reale che fa parte di un 'ontologia (sul libro si parla di WIKIPEDIA come ontologia).

Each unique Wikipedia page acts as the unique id for a particular entity. Significa che ogni pagina di wikipedia credo contenga proprio una singola ENTITY.
This task of deciding which Wikipedia page corresponding to an individual is being referred to by a text mention has its own name: wikification (significa quando cerco qualcosa (la mention) e mi ritorna la prima pagina di wikipedia (che e' l'entity) associata a quel qualcosa).

Entity linking is done in (roughly) two stages: **mention detection** and **mention disambiguation**.
We’ll give two algorithms, one simple classic baseline that uses anchor dictionaries and information from the Wikipedia graph structure and one modern neural algorithm.

### Linking based on Anchor Dictionaries and Web Graph
NOTA: UNA MENTION E' UNA SOTTOSTRINGA DELLO SPAN.
Wikification algorithms define the set of entities as the set of Wikipedia pages, so we’ll refer to each Wikipedia page as a unique entity e. TAGME first creates a catalog of all entities (i.e. all Wikipedia pages, removing some disambiguation and other meta-pages) and indexes them in a standard IR engine like Lucene. For each page e, the algorithm computes an in-link count in(e): the total number of in-links from other Wikipedia pages that point to e. These counts can be derived from Wikipedia dumps. Finally, the algorithm requires an anchor dictionary.

An anchor dictionary  lists for each Wikipedia page, its anchor texts: the hyperlinked spans of text on other pages that point to it.
We compute a Wikipedia anchor dictionary by including, for each Wikipedia page e, e’s title as well as all the anchor texts from all Wikipedia pages that point to e. For each anchor string a we’ll also compute its total frequency freq(a) in Wikipedia (including non-anchor uses), the number of times a occurs as a link (which we’ll call link(a)), and its link probability linkprob(a) = link(a)/freq(a).

**Mention DETECTION** Given a question (or other text we are trying to link), TAGME detects mentions by querying the anchor dictionary for each token sequence up to 6 words. This large set of sequences is pruned with some simple heuristics (for example pruning substrings if they have small linkprobs). The question: When was Ada Lovelace born? might give rise to the anchor Ada Lovelace and possibly Ada, but substrings spans like Lovelace might be pruned as having too low a linkprob, and but spans like born have such a low linkprob that they would not be in the anchor dictionary at all.
**Mention DISAMBIGUATION** If a mention span is unambiguous (points to only one entity/Wikipedia page), we are done with entity linking! However, many spans are ambiguous, matching anchors for multiple Wikipedia entities/pages.
The TAGME algorithm uses two factors for disambiguating ambiguous spans, which have been referred to as prior probability and relatedness/coherence

The FISRT FACTOR is  p(e|a), the prior probability with which the span a refers to a particular entity e.
For each page e ∈ E(a), the probability p(e|a) that anchor a points to e, is the ratio of the number 
of links into e with anchor text a to the total number of occurrences of a as an anchor.

lets see one problem though.
**What Chinese Dynasty came before the Yuan?**
The most common association for the span Yuan in the anchor dictionary is the name of the Chinese currency, i.e., the probability p(Yuan currency| yuan) is very high. Rarer Wikipedia associations for Yuan include the common Chinese last name, a language spoken in Thailand, and the correct entity in this case, the name of the Chinese dynasty. So if we chose based only on p(e|a) , we would make the wrong disambiguation and miss the correct link, Yuan dynasty.

To help in just this sort of case, TAGME uses a second factor, the relatedness of this entity to other entities in the input question. In our example, the fact that the question also contains the span Chinese Dynasty, which has a high probability link to the page Dynasties in Chinese history, ought to help match Yuan dynasty. There is a way of computing it that is really complicated.

### Linking based on Neural-Graph
**More recent entity linking models are based on biencoders, encoding a candidate mention span, encoding an entity, and computing the dot product between the encodings**.
This allows embeddings for all the entities in the knowledge base to be precomputed and cached.

Let’s sketch the ELQ linking algorithm of Li et al. (2020), which is given a question q and a set of candidate entities from Wikipedia with associated Wikipedia text, and outputs tuples (e,ms ,me) of entity id, mention start, and mention end

Fig. 14.13 shows, it does this by encoding each Wikipedia entity using text from Wikipedia, encoding each mention span using text from the question, and computing their similarity, as we describe below
![[neural-graph-entity-linking.png]]
Guarda come ogni entity viene encodata prendendo tutto il testo di quell'entity. Il suo encoding si trova nel CLS token. Viene poi fatto il dot product tra l'encoding dell'entity e quello di una parola nello span e quella sarebbe la probabilita' che quella e' la giusta entity data laMENTION (quindi una parola o un insieme di parole che e' mappabile a una pagina di wikipedia).
Si calcola anche per ogni parola nello span la probabilita' che quella parola sia rilevante (faccia parte della MENTION) data tutta la domanda Q.

**Mention DETECTION**
To get an h-dimensional embedding for each question token, the algorithm runs the question through BERT in the normal way: \[q1 ···qn\] = BERT(\[CLS\]q1 ···qn\[SEP\])

It then computes the likelihood of each span [i, j] in q being an entity mention, in a way similar to the span-based algorithm we saw for the reader above. First we compute the score for i/ j being the start/end of a mention: 
s<sub>start</sub>(i) = w<sub>start</sub> ·q<sub>i</sub> , 
s<sub>end</sub>(j) = w<sub>end</sub> ·q<sub>j</sub> ,
where wstart and wend are vectors learned during training. Next, another trainable embedding, wmention is used to compute a score for each token being part of a mention:
s<sub>mention</sub>(t) = w<sub>mention</sub> ·q<sub>t</sub>.
Mention probabilities are then computed by combining these three scores
![[mentionProba.png]]
COME SI FA ADESSO L'ENTITY LINKING? beh facile
To link mentions to entities, we next compute embeddings for each entity in the set E = e1,··· , ei ,··· , ew of all Wikipedia entities. For each entity ei we’ll get text from the entity’s Wikipedia page, the title t(ei) and the first 128 tokens of the Wikipedia page which we’ll call the description d(ei). This is again run through BERT, taking the output of the CLS token BERT[CLS] as the entity representation: xe = BERT<sub>[CLS]</sub>(\[CLS\]t(e<sub>i</sub>)\[ENT\]d(ei)\[SEP\])

Mention spans can be linked to entities by computing, for each entity e and span [i, j], the dot product similarity between the span encoding (the average of the token embeddings) and the entity encoding. Finally, we take a softmax to get a distribution over entities for each span

TRAINING 
The ELQ mention detection and entity linking algorithm is fully supervised. This means, unlike the anchor dictionary algorithms from Section 14.3.1, it requires datasets with entity boundaries marked and linked.
![[entityLoss.png]]

## Knowledge Based QA
E' il secondo paradigma che utilizza ENTITY LINKING
We use the term knowledge-based question answering for the idea of answering a natural language question by mapping it to a query over a structured database.

Two common paradigms are used for **knowledge-based QA**. The first, **graphbased QA**, models the knowledge base as a graph, often with entities as nodes and relations or propositions as edges between nodes. The second, **QA by semantic parsing**. Both of these methods require some sort of entity linking that we described in the prior section.

### Graph based QA
We’ll focus on the very simplest case of graph-based QA, in which the dataset is a set of factoids in the form of RDF triples, and the task is to answer questions about one of the missing arguments.

	subject      predicate    object 
	Ada Lovelace birth-year   1815
Let’s assume we’ve already done the stage of entity linking introduced in the prior section. Thus we’ve mapped already from a textual mention like Ada Lovelace to the canonical entity ID in the knowledge base.
For simple triple relation question answering, the next step is to determine **which relation is being asked about**, mapping from a **string** like “When was ... born” to **canonical relations** in the knowledge base like birth-year.
NOTA: possono esserci piu relations dentro una domanda. Quindi ci sono algoritmi per relation detection.
For simple questions, where we assume the question has only a single relation, relation detection and linking can be done in a way resembling the neural entity linking models: computing similarity (generally by dot product) between the encoding of the question text and an encoding for each possible relation.

Most algorithms have a final stage which takes the top j entities and the top k relations returned by the entity (quindi al termine dell'entity linking step) and relation inference steps,searches the knowledge base for triples containing those entities and relations, and then ranks those triples.
This ranking can be heuristic.
### QA by semantic parsing

The second kind of knowledge-based QA uses a semantic parser to map the question to a structured program to produce an answer. The logical form of the question is  either in the form of a query or can easily be converted into one.
Semantic parsing algorithms can be supervised fully with questions paired with a hand-built logical form, or can be weakly supervised by questions paired with an answer (the denotation), in which the logical form is modeled only as a latent variable.
The task is then to take those pairs of training tuples and produce a system that maps from new questions to their logical forms. A common baseline algorithm is a simple sequence-to-sequence model, for example using BERT to represent question tokens, passing them to an encoder-decoder.
FINE

## QA with Large language models
An alternative approach to doing QA is to query a pretrained language model, forcing a model to answer a question solely from information stored in its parameters (tipo CHATGPT).
Language modeling is not yet a complete solution for question answering; for example in addition to not working quite as well, they suffer from poor interpretability (unlike standard QA systems, for example, they currently can’t give users more context by telling them what passage the answer came from).

## CLASSIC QA MODELS
While neural architectures are the state of the art for question answering, pre-neural architectures using hybrids of rules and feature-based classifiers can sometimes achieve higher performance.
Here we summarize one influential classic system, the Watson DeepQA system from IBM that won the Jeopardy! Two phrases examples:
1. Poets and Poetry: **He** was a bank clerk in the Yukon before he published “Songs of a Sourdough” in 1907.
2. THEATRE: A new play based on this **Sir Arthur Conan Doyle canine** classic opened on the London stage in 2007
![[classical-QA-model.png]]
### QUESTION PROCESSING
In this stage the questions are parsed, named entities are extracted (Sir Arthur Conan Doyle identified as a PERSON, Yukon as a GEOPOLITICAL ENTITY, “Songs of a Sourdough” as a COMPOSITION), coreference is run (he is linked with clerk). 
The question focus, shown in bold in both examples, is extracted. The focus is the string of words in the question that corefers with the answer. It is likely to be replaced by the answer in any answer string found and so can be used to align with a supporting passage. It is extracted by rules.
The lexical answer type (shown in blue above) is a word or words which tell us something about the semantic type of the answer.These lexical answer types are again extracted by rules.
Finally the question is classified by type (definition question, multiple-choice, puzzle, fill-in-the-blank). This is generally done by writing pattern-matching regular expressions over words or parse trees.
NON VADO NEL DETTAGLIO. LEGGI PARAGRAFO e' anche fuorviante sto capitolo imo

## Evaluation of Factoid Answers
Factoid question answering is commonly evaluated using mean reciprocal rank, or MRR. MRR is designed for systems that return a short ranked list of answers or passages for each test set question, which we can compare against the (human-labeled) correct answer. First, each test set question is scored with the reciprocal of the rank of the first correct answer. For example if the system returned five answers to a question but the first three are wrong (so the highest-ranked correct answer is ranked fourth), the reciprocal rank for that question is 1/4. The score for questions that return no correct answer is 0. The MRR of a system is the average of the scores for each question in the test set.
FINE