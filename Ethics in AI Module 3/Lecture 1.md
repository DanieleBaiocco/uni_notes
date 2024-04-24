Posso integrare la knowledge degli esperti dentro un modello data-driven
Questa cosa e' molto utile perche' mi permette anche di semplificare il learning process (se ho una migliore comprensione del problema e di cio' che serve, sicuramente e' tutto molto piu' veloce).
AI puo' essere usata per applicazioni quali: *autonomous weapons*, *social scoring* e *fake news* che influenzano l'opinione pubblica.

# Iniziative
## Eccesso di Iniziative
Ogni nuova iniziativa fornisce un insieme di principi e valori nuovi, o cerca di guidare la crescita e adozione dell'Intelligenza Artificiale. C'e' pero' un numero eccessivo di queste iniziative che ha come pericolo quello di portare a un sovraccarico e confusione.

... *aggiungere lista di iniziative e principi formulati dagli stati*...
In tutto fanno 47 principi.
## A Unied Framework of Five Principles for AI in Society
Si parla poi del fatto che e' tutto riassumibile in 4 principi + 1(guardati il paper A Unied Framework of Five Principles for AI in Society - MANDATORY READING che sta in Ethics in AI Module 1 - READINGS).
Di base questi sono i seguenti:
1. **beneficence, 
2. **non-maleficence,**
3. **autonomy,**
4.  **justice**.
5. Explicability (aggiunto poi, non era presente nel riassunto di tutti i trattati usciti sulla AI)

# Ethical Guidelines of Trustworthy AI - EGTAI

Trustworthy AI has three components (per quelli che hanno scritto sto paper ovviamente), which should be met throughout the system's entire life cycle:
1. it should be **lawful**, complying with all applicable laws and regulations
2. it should be **ethical**, ensuring adherence to ethical principles and values and 
3. it should be **robust**, both from a technical and social perspective since, even with good intentions, **AI systems can cause unintentional harm**.
These Guidelines set out a framework for achieving Trustworthy AI. The framework does not explicitly deal with Trustworthy AI’s first component (lawful AI). Instead, it aims to offer guidance on the second and third components: fostering and securing ethical and robust AI.
 There are 3 chapters:
 1. Chapter I identifies the ethical principles and their correlated values that must be respected in the development, deployment and use of AI systems.
 2. Drawing upon Chapter I, Chapter II provides guidance on how Trustworthy AI can be realised, by listing **seven** requirements that AI systems should meet during development and deployment. Both technical and non-technical methods can be used for their implementation. They are: 
	 1. human agency and oversight: legato a **autonomy**, 
	 2. technical robustness and safety, legato a **nonmaleficence**
	 3. privacy and data governance: legato a **nonmaleficence**, 
	 4. transparency: legato a **explicability**,
	 5. diversity, non-discrimination and fairness: legato a **justice**, 
	 6. environmental and societal well-being: legato a **nonmaleficence**, 
	 7. accountability: legato a **explicability**.
 3. Chapter III provides a concrete and non-exhaustive Trustworthy AI assessment list aimed at operationalising the key requirements set out in Chapter II. This assessment list will need to be tailored to the specific use case of the AI system.

Il motivo di questo paper risiede nell'evitare che vengano utilizzate delle AI pericolose, che possano mettere a rischio l'umanita'.
Esempi sono mostrati nelle figure qua sotto:
![[ethics3.png]]
![[ethics31.png]]

Per adesso la situa e' questa:![[situazione 1.png]]
In cui l'interpretability e' inversamente proporzionale all'accuracy. E si vuole provare a aumentare l'interpretabilita' di questi modelli, soprattutto degli ensembles e delle DNNs.
Ho poi anche questo altro tradeoff tra Robustness del modello (un modello e' robusto quando performa bene anche quando i test exaples sono diversi dalla training distribution) e l'accuracy. 
![[Pasted image 20240321160855.png]]
Quindi si hanno diversi tradeoffs tra accuracy e interpretability, tra accuracy e fairness (plot che non e' stato inserito), tra accuracy e robustness. 

### EGTAI- Definizioni e esempi
Riguardo ai 7 principi di prima, per ognuno si hanno cose dette a riguardo ovviamente.
### Human agency and oversight
1. Per quanto riguarda la **human agency** si vuole far questo:
	**empower** human beings, **allowing them to make informed decisions** and fostering their fundamental rights. Agency may be achieved through governance mechanisms such as a human-centric approaches, AI for social good, human computation, interactive machine learning
2. Per quanto riguarda la **human oversight** (ovvero la supervisione umana) si vuole fare questo:
	Ensuring that an AI system **does not undermine human autonomy** or cause other adverse effects. Oversight may be achieved through **governance mechanisms**such as a human-in-the-loop, human-on-the-loop, human-in-command
### Technical robustness and safety
Qui si dice la seguente cosa riguardante la **sicurezza di un AI SYSTEM**:
Prevention of unintentional harm (physical or psychological) especially to human beings, and by extension to other material or immaterial elements that may be valuable for humans, including the system itself and the minimization of the consequences of intentional harm. These includes:
1. resilience of AI-based systems (to attacks and security)
2. ensuring fallback plans (in case something goes wrong)
3. general safety, and being accurate, reliable and reproducible
4. cover the way and conditions in which the system ceases its operation, and the consequences of stopping
Riguardo alla **ROBUSTNESS** invece viene detta la seguente cosa:
Emphasises that safety and —conditionally to it— functionality, must be **preserved** in different situations and also under **harsh conditions**, including *unanticipated errors, exceptional situations, unintended or intended damage, manipulation or catastrophic states*.

Riguardo alla **REPRODUCIBILITY** che era stata nominata prima si dice questo:
Once robustness and safety have been addressed, an important dimension in this key requirement for trustworthy AI is reproducibility. Per capire questa parola dobbiamo metterla a confronto con parole simili:
1. *Repeatability* (same team, same experimental setup): which means that an individual or a team of individuals can reliably repeat his/her/their own experiment
2. *Replicability* (different team, same experimental setup): an independent group of individuals can obtain the same result using artifacts that they independently develop in their entirety
3. *Reproducibility* (different team, different experimental setup with stated precision): a different independent group can obtain the same result using their own artifacts.
Alcuni esempi pratici sono:
1. sicurezza nelle self-driving cars: Safety concerns driving without accidents neither for the driver nor for other vehicles, pedestrians, bicycles. Self-driving cars should be safe even under conditions as fog, rain, snow, heavy traffic
2. sicurezaza nelle NN: DNN can be unstable, infact by applying some perturbations to inputs we can arbitrarily change the output (ci sono casi in cui un'immagine di un gatto, se perturbata con del noise, viene classificata come l'immagine di un tostapane)
### Privacy and data governance
This is the most well-known and better-regulated aspect of AI

Pero' c'e' comunque questa cosa che sto per scrivere che dovrebbe spaventare un po' tutti:
`Easily accessible digital records of behavior, Facebook Likes, can be used to automatically and accurately predict a range of highly sensitive personal attributes: sexual orientation, ethnicity, religious and political views, personality traits, intelligence, happiness, use of addictive substances, parental separation, age, and gender`
Questo esperimento e' stato condotto:
1. A dataset of 58,000 volunteers has made available their Facebook Likes and detailed personal data, profiles etc. for machine learning 
2. The learned model is accurate and discriminates among different categories (homosexual and heterosexuality with 88% accuracy, Democratics and Repubblicans with 85% accuracy)
C'e' anche un'altra questione che e' quella legata al *generative AI*. Una generative AI viene trainato con dati reali e genera magari un'immagine sintetica che e' simile al training set. La domanda e': **Di chi e' l'immagine generata?** Perche'  e' di base frutto di altre immagini (di cui non si e' chiesto il consenso tra l'altro) che costituivano il dataset.

### Transparency
Ensure appropriate information reaches the relevant stakeholders. Humans must be informed of systems’ capabilities and limitations and always be aware that they are interacting with AI systems.
Explanations should be timely, adapted and communicated to the stakeholder audience concerned

Not common definition of explanation: this is the most widely accepted:
*An explanation is the evidence, support, or reasoning related to a **system’s output or process,** where the output of a system differs by task, and the process refers to the procedures, design, and system workflow which underlie the system.*
'
The explanations returned depend on various factors, such as:
1. the type of task they are needed for
2. on which kind of data the AI system acts
3. who is the final user of the explanation
4. if they allow (se i final users, penso si riferisca a loro) to explain the whole behavior of the AI system (global explanation) or reveal the reasons for the decision only for a particular instance (local explanation)'
5. the business perspective, i.e., which are the implications of companies in having explainable and interpretable systems and models, in terms of business strategies and secrecy (cioe' magari non e' possibile spiegare una determinata cosa perche' un segreto dell'azienda immagino? o comunque fa parte di una strategia di mercato dell'azienda)
6. the fact that, in a decentralized node, an explanation could require information that is not directly available on site (ci puo' stare pure questo) 
![[explanation.png]]
come si vede da sto esempio, l'explanation e' importantissima anche per dire il motivo di un determinato outcome. E' proprio un qualcosa che mi permette di comprendere l'outcome dell'AI.

### Diversity, non-discrimination and fairness
La diversity e la non-discrimination vengono spiegate nel contesto dell'AI nel paper nel seguente modo: consistono nel
1. inclusion of diverse data and people, and ensures that individuals at risk of exclusion have equal access to AI benefits
2. diversity, **it advocates for the need for heterogeneous and randomly sampling procedures for data acquisition, diverse representation of a population that includes minorities, and the assurance for non-discriminating automated processes that lead to unfairness or biased models**

La definizione di **fairness** secondo Article 21 of the EU Charter of Fundamental Rights e' la seguente:
*any discrimination based on any ground such as sex, race, colour, ethnic or social origin, genetic features, language, religion or belief, political or any other opinion, membership of a national minority, property, birth, disability, age or sexual orientation shall be prohibited.*

They describe two different discrimination scenarios:
1. *direct discrimination* (disparate treatment)
2. *indirect discrimination* (disparate impact): when a seemingly “neutral provision, criterion or practice” disproportionately disadvantages members of a given sensitive group compared to others
Il paper parla anche di **computational fairness** che consiste in *potential biases and discrimination that can arise from the use of computational algorithms*. Nel paper si vuole fare in modo che *algorithms do not perpetuate or amplify existing biases and do not discriminate against certain groups of people based on sensitive attributes*
Si parla anche di **fairness metrics** ovvero di  *Quantitative measurement used to assess and quantify the fairness or bias of an algorithm’s predictions or decisions*  (le abbiamo viste in AI IN INDUSTRY).![[fairness.png]]

### Environmental and societal well-being
*AI-based systems should benefit all humankind, not only at the present time but also in future generations. Therefore, AI-based systems must be sustainable and environmentally friendly, so that the technological adoption of AI does not entail a progressive depletion of natural resources and maintains an ecological balance*

Il problema di base e' che *AI needs High performance computing that has a heavy carbon footprint*

Pero' overall la AI puo' aiutare nel **ridurre** IL CARBON FOOTPRINT tramite la sua applicazione in diversi contesti aziendali facendo:
1. Power aware management
2. Optimal allocation and scheduling
3. Cooling optimization and thermal-aware workload dispatching
4. Anomaly detection
Quindi e' worthit

AI can improve **social welfare** nei seguenti modi:
1. performing routine tasks in an autonomous safer, and more efficient fashion, enhancing productivity and improving the quality of life of **humankind**
2. speeding up processes, smoothing administrative bottlenecks and saving paperwork
3. helping city planners, e.g., by visualizing the consequences of climate change, predicting future floods

### Accountability
ensure responsibility and accountability for the development, deployment, maintenance and/or use of AI systems and their outcomes

If AI produces harm, understand who is responsible for it:
1. Writer of the algorithm
2. Owner of the object in which AI algorithm works (the car for example)
3. The human supervising it
Sto paper risponde alle seguenti domande :
1. Health: What if diagnoses are wrong because of issues with ML and A.I.?
2. Transportation: Let us consider autonomous vehicles and whether the algorithmic application fails. Who is accountable for A.I. damages to stakeholders?
3. ecc..

C'e' l'**auditability** che fa parte dell'accoountability che e' 
1. development of practical tools capable of verifying desirable properties of neural networks such as stability , sensitivity , relevance or reachability 
2. as well as metrics beyond explainability, such as traceability , data quality and integrity

Auditability is becoming increasingly important when standards are being materialized touching upon all AI requirements: IEEE, ISO/IEC and CEN/CENELEC, which are implementing concrete guidelines to apply trustworthy AI requirements in industrial setups

# AI for social Good
L'AI puo' aiutare la societa' nel *social good*: in pratica per aiutare nel ridurre, mitigare o eradicare un problema sociale o di ambiente senza introdurre nuovi pericoli o senza amplificare pericoli gia' esistenti.
 In particolare c'e' un progetto/paper che si chiama AI4SG project che parla proprio di questo , che ha come obiettivo proprio quello di *Design, development, and deployment of AI systems in ways that prevent, mitigate, or resolve problems adversely affecting human life and/or the well-being of the natural world, and/or enable socially preferable and/or environmentally sustainable developments*

## Fattori rilevanti per fare design/usare AI per il Social Good

C'e' una lista di fattori rilevanti da dover mettere in pratica per ottenere una *AI per il Social Good*:
1. Falsifiability and incremental deployment: AI4SG designers should identify *falsifiable* (falsificabile) requirements and test them in incremental steps from the lab to the ‘outside world’
2. safeguards against the manipulation of predictors: AI4SG designers should adopt safeguards that (i) ensure non-causal indicators do not inappropriately skew interventions, and when appropriate, limit knowledge of how inputs affect outputs from AI4SG systems to prevent manipulation.
3. receiver-contextualized intervention: AI4SG designers should build decision-making systems in consultation with users interacting with (and impacted by) these systems; with an understanding of user characteristics, the methods of coordination, and the purposes and effects of an intervention; and with respect for the user’s right to ignore or modify interventions.
4. receiver-contextualized explanation and transparent purposes: AI4SG designers should first choose an LoA for AI explanation that fulfils the desired explanatory purpose and is appropriate to both the system and receivers. Then, designers should deploy arguments that are rationally and suitably persuasive for the receivers to deliver the explanation. Finally, designers should ensure that the goal (the system’s purpose) for which an AI4SG system is developed and deployed is knowable to the receivers of its outputs by default.
5. privacy protection and data subject consent: AI4SG designers should respect the threshold of consent established for processing datasets of personal data.
6. situational fairness: AI4SG designers should remove from relevant datasets variables and proxies that are irrelevant to an outcome, except when their inclusion supports inclusivity, safety, or other ethical imperatives.
7. human-friendly semanticization: AI4SG designers should not hinder (ostacolare) the ability of people to semanticize (that is, to give meaning to, and make sense of) something.