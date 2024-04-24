Part of speech tagging can tell us that words like Janet, Stanford University, and Colorado are all proper nouns; being a proper noun is a grammatical property of these words. But viewed from a semantic perspective, these proper nouns refer to different kinds of entities: Janet is a person, Stanford University is an organization, and Colorado is a location.
A named entity is, roughly speaking, anything that can be referred to with a proper name: a person, a location, an organization.
The task of named entity recognition (NER) is to find spans of text that constitute proper names and tag the type of the entity.
Four entity tags are most common: PER (person), LOC (location), ORG (organization), or GPE (geo-political entity). However, the term named entity is commonly extended to include things that aren’t entities per se, including dates, times, and other kinds of temporal expressions, and even numerical expressions like prices.

Unlike part-of-speech tagging, where there is no segmentation problem since each word gets one tag, the task of named entity recognition is to find and label spans of text, and is difficult partly because of the ambiguity of segmentation; we need to decide what’s an entity and what isn’t, and where the boundaries are. Indeed, most words in a text will not be named entities. Another difficulty is caused by type ambiguity. The mention JFK can refer to a person, the airport in New York, or any number of schools, bridges, and streets around the United States.

The standard approach to sequence labeling for a span-recognition problem like NER is BIO tagging
![[biotagging.png]]
In BIO tagging we label any token that begins a span of interest with the label B, tokens that occur inside a span are tagged with an I, and any tokens outside of any span of interest are labeled O. While there is only one O tag, we’ll have distinct B and I tags for each named entity class. The number of tags is thus 2n+1 tags, where n is the number of entity types.

There are three approaches to NER: rule based, feature based, neural
## Feature Based
![[esempioFeatureBasedNER.png]]
nel feature based ci metto che tipo di parola e'  (NNP, NP, ...) quindi postaggo. Poi anche se e' una short sequence, se e' una nel gazetteer, ecc... come riportato in figura fondamentalmente. E poi faccio la classificazione predictando il target BIO date le features delle parole intorno e i targets delle parole precedenti.

Il neural based e' easy, praticamente si parla di neural networks ovviamente. Il rule based e' palloso.