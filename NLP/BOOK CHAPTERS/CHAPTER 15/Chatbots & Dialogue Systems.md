**TURNS** A dialogue is a sequence of turns (C1, A2, C3, and so on), each a single contribution from one speaker to the dialogue(as if in a game: I take a turn, then you take a turn then me, and so on).
Turn structure has important implications for spoken dialogue. A system has to know when to stop talking.
**SPEECH ACTS** A key insight into conversation  is that each utterance in a dialogue is a kind of action being performed by the speaker. These actions are commonly called speech acts.
**Constatives**: committing the speaker to something’s being the case (answering, claiming, confirming, denying, disagreeing, stating)
**Directives**: attempts by the speaker to get the addressee to do something (advising, asking, forbidding, inviting, ordering, requesting)
**Commissives**: committing the speaker to some future course of action (promising, planning, vowing, betting, opposing)
**Acknowledgments**: express the speaker’s attitude regarding the hearer with respect to some social action (apologizing, greeting, thanking, accepting an acknowledgment)

A user asking a person or a dialogue system to do something (‘Turn up the music’) is issuing a DIRECTIVE. Asking a question that requires an answer is also a way of issuing a DIRECTIVE: in a sense when the system says (A2) “what day in May did you want to travel?” it’s as if the system is (very politely) commanding the user to answer. By contrast, a user stating a constraint (like C1 ‘I need to travel in May’) is issuing a CONSTATIVE. A user thanking the system is issuing an ACKNOWLEDGMENT. The speech act expresses an important component of the intention of the speaker (or writer) in saying what they said.

**GROUNDING** A dialogue is not just a series of independent speech acts, but rather a collective act performed by the speaker and the hearer. Like all collective acts, it’s important for the participants to establish what they both agree on, called the common ground. Speakers do this by grounding each other’s utterances. Grounding means acknowledging that the hearer has understood the speaker; like an ACK used to confirm receipt in data communications.Humans constantly ground each other’s utterances. We can ground by explicitly saying “OK”, as the agent does in A8 or A10. Or we can ground by repeating what the other person says; in utterance A2 the agent repeats “in May”, demonstrating her understanding to the client.

**DIALOG STRUCTURE** 
Consider, for example, the local structure between speech acts discussed in the field of conversational analysis
QUESTIONS set up an expectation for an ANSWER. PROPOSALS are followed by ACCEPTANCE (or REJECTION). COMPLIMENTS (“Nice jacket!”) often give rise to
DOWNPLAYERS (“Oh, this old thing?”). These pairs, called adjacency pairs are composed of a first pair part and a second pair part and these expectations can help systems decide what actions to take. However, dialogue acts aren’t always followed immediately by their second pair part. The two parts can be separated by a side sequence or subdialogue. 

**INITIATIVE** Sometimes a conversation is completely controlled by one participant. For example a reporter interviewing a chef might ask questions, and the chef responds. We say that the reporter in this case has the conversational initiative. In normal human-human dialogue, however, it’s more common for initiative to shift back and forth between the participants, as they sometimes answer questions, sometimes ask them, sometimes take the conversations in new directions, sometimes not.You may ask me a question, and then I respond asking you to clarify something you said, which leads the conversation in all sorts of ways. We call such interactions mixed initiative.
Mixed initiative, while the norm for human-human conversations, is very difficult for dialogue systems to achieve. It’s much easier to design dialogue systems to be passive responders. 

**INFERENCE** . Implicature means a particular class of licensed inferences.

These subtle characteristics of human conversations (**turns, speech acts, grounding, dialogue structure, initiative, and implicature**) are among the reasons it is difficult to build dialogue systems that can carry on natural conversations with humans. Many of these challenges are active areas of dialogue systems research.

# Chatbots
The simplest kinds of dialogue systems are chatbots, systems that can carry on extended conversations with the goal of mimicking the unstructured conversations or ‘chats’ characteristic of informal human-human interaction.
Like practically everything else in language processing, chatbot architectures fall into two classes: rule-based systems and corpus-based systems.
## Rulebased chatbots
A few years after ELIZA, another chatbot with a clinical psychology focus, PARRY (Colby et al., 1971), was used to study schizophrenia. In addition to ELIZAlike regular expressions, the PARRY system included a model of its own mental state, with affect variables for the agent’s levels of fear and anger; certain topics of conversation might lead PARRY to become more angry or mistrustful. If PARRY’s anger variable is high, he will choose from a set of “hostile” outputs. If the input mentions his delusion topic, he will increase the value of his fear variable and then begin to express the sequence of statements related to his delusion. Parry was the first known system to pass the Turing test (in 1972!); psychiatrists couldn’t distinguish text transcripts of interviews with PARRY from transcripts of interviews with real paranoids
## Corpusbased chatbots
Corpus-based chatbots, instead of using hand-built rules, mine conversations of human-human conversations. These systems are enormously data-intensive, requiring hundreds of millions or even billions of words for training.
Most corpus based chatbots produce their responses to a user’s turn in context either by **retrieval methods** (using information retrieval to grab a response from some corpus that is appropriate given the dialogue context) or **generation methods** (using a language model or encoder-decoder to generate the response given the dialogue context).
In either case, systems mostly generate a single response turn that is appropriate given the entire conversation so far (for conversations that are short enough to fit into a single model’s window).
### Response by retrieval
The retrieval method of responding is to think of the user’s turn as a query q, and our job is to retrieve and repeat some appropriate turn r as the response from a corpus of conversations C. Generally C is the training set for the system, and we score each turn in C as a potential response to the context q selecting the highest-scoring one. The scoring metric is similarity: we choose the r that is most similar to q, using any of the IR methods we saw in Section 14.1. 
- This can be done using classic IR techniques to compute tf-idf models for C and q, choosing the r that has the highest tf-idf cosine with q.
- Another version of this method is to return the response to the turn resembling q; that is, we first find the most similar turn t to q and then return as a response the following turn r.
- Alternatively, we can use the neural IR techniques . The simplest of those is a bi-encoder model, in which we train two separate encoders, one to encode the user query and one to encode the candidate response, and use the dot product between these two vectors as the score . For example to implement this using BERT, we would have two encoders BERT<sub>Q</sub> and BERT<sub>R</sub> and we could represent the query and candidate response as the \[CLS\] token of the respective encoders:
 ![[IRinRetrievalAnswer.png]]
	 The IR-based approach can be extended in various ways, such as by using more sophisticated neural architectures (Humeau et al., 2020), or by using a longer context for the query than just the user’s last turn, up to the whole preceding conversation. Information about the user or sentiment or other information can also play a role
### Response by generation 
An alternate way to use a corpus to generate dialogue is to think of response production as an encoder-decoder task— transducing from the user’s prior turn to the system’s turn.

# GUS: Simple Frame-based Dialogue Systems
We turn now to task-based dialogue, in which a dialogue system has the goal of helping a user solve some task like making an airplane reservation or buying a product.
All modern task-based dialogue systems, whether the simple GUS architecture we describe here, or the more sophisticated dialogue state architectures we turn to in the following section, are based around frames. A frame is a kind of knowledge structure representing the kinds of intentions the system can extract from user sentences, and consists of a collection of slots, each of which can take a set of possible values. Together this set of frames is sometimes called a domain ontology.

The set of slots in a task-based dialogue frame specifies what the system needs to know, and the filler of each slot is constrained to values of a particular semantic type. In the travel domain, for example, a slot might be of type city (hence take on values like San Francisco, or Hong Kong) or of type date, airline, or time
## Control structure for frame-based **dialogue**
The control architecture for frame-based dialogue systems is designed around the frame. The system’s goal is to fill the slots in the frame with the fillers the user intends, and then perform the relevant action for the user (answering a question, or booking a flight).To do this, the system asks questions of the user (using pre-specified question templates associated with each slot of each frame, as shown in Fig. 15.10), filling any slot that the user specifies. If a user’s response fills multiple slots, like the following: (15.4) I want a flight from San Francisco to Denver one way leaving after five p.m. on Tuesday. the system fills all the relevant slots, and then continues asking questions to fill the remaining slots, skipping questions associated with filled slots.
The GUS architecture also has condition-action rules attached to slots. For example, a rule attached to the DESTINATION slot for the plane booking frame, once the user has specified the destination, might automatically enter that city as the default StayLocation for the related hotel booking frame.
Or if the user specifies the DESTINATION DAY for a short trip the system could automatically enter the ARRIVAL DAY.

Because of this need to dynamically switch control, the GUS architecture is a production rule system. Different types of inputs cause different productions to fire, each of which can flexibly fill in different frames. The production rules can then switch control according to factors such as the user’s input and some simple dialogue history like the last question that the system asked. Once the system has enough information it performs the necessary action (like querying a database of flights) and returns the result to the user.

## Determining Domain, Intent, and Slot fillers in GUS
The goal of the next component in the frame-based architecture is to extract three things from the user’s utterance. The first task is domain classification: is this user for example talking about airlines, programming an alarm clock, or dealing with their calendar? Of course this 1-of-n classification tasks is unnecessary for single-domain systems that are focused on, say, only calendar management, but multi-domain dialogue systems are the modern standard. The second is user intent determination: what general task or goal is the user trying to accomplish? For intent determination example the task could be to Find a Movie, or Show a Flight, or Remove a Calslot filling endar Appointment. Finally, we need to do slot filling: extract the particular slots and fillers that the user intends the system to understand from their utterance with respect to their intent. 

# The Dialogue-State architecture
Like the GUS systems, the dialogue-state architecture has a component for extracting slot fillers from the user’s utterance, but generally using machine learning rather than rules.(This component is sometimes called the NLU or **SLU** component, for ‘Natural Language Understanding’, or ‘Spoken Language Understanding’, using the word “understanding” loosely.)The **dialogue state tracker** maintains the current state of the dialogue (which include the user’s most recent dialogue act, plus the entire set of slot-filler constraints the user has expressed so far).The **dialogue policy** decides what the system should do or say next. The dialogue policy in GUS was simple: ask questions until the frame was full and then report back the results of some database query. But a more sophisticated dialogue policy can help a system decide when to answer the user’s questions, when to instead ask the user a clarification question, when to make a suggestion, and so on. n. Finally, dialogue state systems have a **natural language generation component**. In GUS, the sentences that the generator produced were all from pre-written templates. But a more sophisticated generation component can condition on the exact context to produce turns that seem much more natural.

## Dialogue acts
Dialogue-state systems make use of dialogue acts. Dialogue acts represent the interactive function of the turn or sentence, combining the idea of speech acts and grounding into a single representation. Different types of dialogue systems require labeling different kinds of acts, and so the tagset—defining what a dialogue act is exactly— tends to be designed for particular tasks.
**SPEECH ACTS** A key insight into conversation  is that each utterance in a dialogue is a kind of action being performed by the speaker. These actions are commonly called speech acts.
![[tagsetExample.png]]
![[tagsetFilled.png]]
Figure 15.12 shows a tagset for a restaurant recommendation system, and Fig. 15.13 shows these tags labeling a sample dialogue from the HIS system (Young et al., 2010). This example also shows the content of each dialogue acts, which are the slot fillers being communicated. So the user might INFORM the system that they want Italian food near a museum, or CONFIRM with the system that the price is reasonable

## Slot filling
The task of slot-filling, and the simpler tasks of domain and intent classification, are special cases of the task of supervised semantic parsing discussed in Chapter 20, in which we have a training set that associates each sentence with the correct set of slots, domain, and intent.
A simple method is to train a sequence model to map from input words representation to slot fillers, domain and intent. For example given the sentence:
I want to fly to San Francisco on Monday afternoon please 
we compute a sentence representation, for example by passing the sentence through a contextual embedding network like BERT. The resulting sentence representation can be passed through a feedforward layer and then a simple 1-of-N classifier to determine that the domain is AIRLINE and and the intent is SHOWFLIGHT.

Our training data is sentences paired with sequences of BIO labels:
O O     O  O   O  B-DES   I-DES         O    B-DEPTIME  I-DEPTIME  O 
I  want to fly  to San        Francisco  on   Monday      afternoon     please
Recall from Chapter 8 that in BIO tagging we introduce a tag for the beginning (B) and inside (I) of each slot label, and one for tokens outside (O) any slot label.
Fig. 15.14 shows the architecture. The input is a series of words w1...wn, which is passed through a contextual embedding model to get contextual word representations. This is followed by a feedforward layer and a softmax at each token position over possible BIO tags, with the output a series of BIO tags s1...sn. We can also combine the domain-classification and intent-extraction tasks with slot-filling simply by adding a domain concatenated with an intent as the desired output for the final EOS token
![[slotfilling.png]]

## Dialogue State Tracking
The job of the dialogue-state tracker is to determine both the current state of the frame (the fillers of each slot), as well as the user’s most recent dialogue act. The dialogue-state thus includes more than just the slot-fillers expressed in the current sentence; it includes the entire state of the frame at this point, summarizing all of the user’s constraints.

## Dialogue policy
The goal of the dialogue policy is to decide what action the system should take next, that is, what dialogue act to generate.
More formally, at turn i in the conversation we want to predict which action Ai to take, based on the entire dialogue state.
These probabilities can be estimated by a neural classifier using neural representations of the slot fillers (for example as spans) and the utterances (for example as sentence embeddings computed over contextual embeddings) More sophisticated models train the policy via reinforcement learning.

C'e' anche altro tipo l'evaluation ecc.. ma e' una roba alquanto semplice 