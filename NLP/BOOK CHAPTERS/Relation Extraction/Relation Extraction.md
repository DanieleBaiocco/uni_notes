This process of information extraction (IE) turns the unstructured information embedded in texts into structured data, for example for populating a relational database to enable further processing.

We begin with the task of relation extraction: finding and classifying semantic relations among entities mentioned in a text, like child-of (X is the child-of Y), or part-whole or geospatial relations.
Relation extraction has close links to populating a relational database, and knowledge graphs, datasets of structured relational knowledge, are a useful way for search engines to present information to users. In the following figure there are examples of RELATIONS we would like to automatically extract.
![[REexamples.png]]
Let’s assume that we have detected the named entities in our sample text (perhaps using the techniques of Chapter 8), and would like to discern the relationships that exist among the detected entities:
Citing high fuel prices, [ORG United Airlines] said [TIME Friday] it has increased fares by [MONEY $6] per round trip on flights to some cities also served by lower-cost carriers. [ORG American Airlines], a unit of [ORG AMR Corp.], immediately matched the move, spokesman [PER Tim Wagner] said. [ORG United], a unit of [ORG UAL Corp.], said the increase took effect [TIME Thursday] and applies to most routes where it competes against discount carriers, such as[LOC Chicago] to [LOC Dallas] and [LOC Denver] to [LOC San Francisco].

![[modelbasedER.png]]
Notice how this model-theoretic view subsumes the NER task as well; named entity recognition corresponds to the identification of a class of unary relations.

Posso fare relation extraction con Rule based , quindi matchando delle regole scritte a mano
Oppure con Supervised Learning
Nel Supervised anche stavolta posso farlo con Feature-based tra due entities o neurale.
Per quello feature based ho un esempio di features qua sotto: 
![[featureBasedER.png]]

Per quanto riguarda quello neurale ho : 
A typical Transformer-encoder algorithm, shown in Fig. 21.7, simply takes a pretrained encoder like BERT and adds a linear layer on top of the sentence representation (for example the BERT [CLS] token), a linear layer that is finetuned as a 1-of-N classifier to assign one of the 43 labels. The input to the BERT encoder is partially de-lexified; the subject and object entities are replaced in the input by their NER tags. This helps keep the system from overfitting to the individual lexical items.

BTW non va sempre bene con SUPERVISED, quindi si sono utilizzati nel tempo approcci di SEMI SUPERVISED tramite bootstrapping:

Supervised machine learning assumes that we have lots of labeled data. Unfortunately, this is expensive. But suppose we just have a few high-precision seed patterns, like those in Section 21.2.1, or perhaps a few seed tuples. That’s enough to bootstrap a classifier! Bootstrapping proceeds by taking the entities in the seed pair, and then finding sentences (on the web, or whatever dataset we are using) that contain both entities. 
