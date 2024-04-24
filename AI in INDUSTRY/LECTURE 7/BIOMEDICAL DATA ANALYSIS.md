I clienti non sono interessati nel creare un PREDICTOR, ma vogliono CAPIRE COSA STA SUCCEDENDO, cosa provoca una certa condizione ecc...
Abbiamo I DATI (in sto caso sono SYNTHETIC DATA, ma e' uguale)
Ogni cosa e' modellata come un vero use-case pero'.
Ho 500 data points, non tanti ma E' LA NORMA in questo campo. Spesso la condizione da analizzare e' molto RARA. Oppure I DATI SONO MOLTO SCATTERED quindi e' difficile mergiarli in un solo dataset.
Ci sono delle features numeriche e categoriche.

LA PRIMA COSA CHE FACCIO E' PLOTTARE I NUMERICAL ATTRIBUTES:
![[distribution7.png]]
Ho che la maggior parte segue una NORMAL DISTRIBUTION
Ho anche NO MISSING VALUES, cosa molto rara per questo tipo di DATASETS.
Nota: dal _describe()_ capisco quali features sono CATEGORICAL (se il min e' 0 e il max e' 1, o 0 e un numero non troppo grande ecc...) e quali sono NUMERICAL. Se non riesco a capirlo dai dati, lo chiedo al CLIENTE COMUNQUE EH, importante sta cosa.
Poi plotto la distribuzione dei CATEGORICAL ATTRIBUTES:![[catdistribution7.png]]
Noto se ci sono degli UNBALANCE nella distribution di sti binary attributes. Qualcuno lo e' altri no.
Controllo poi la target distribution![[targetdistribution7.png]]
E' importante questa cosa, per vedere se il DATASET e' BIASED magari. In sto caso STO A POSTO. E' balanced.

**IMPORTANTE**
Come faccio a vedere come le cose sono LINKED?
1. Fare correlation matrix
2. Se sono particolarmente interessato al LINK tra inputs e OUTPUT allora posso fare se ho NUMERIC TARGET con NUMERIC INPUT faccio SCATTER PLOT
3. Se ho CATEGORICAL INPUT e NUMERIC TARGET allora faccio BOXPLOT
4. Se ho CATEGORICAL TARGET allora POSSO VEDERE LA CLASS DISTRIBUTION per different values dell'INPUT
Faccio il punto 4., mostro quindi i valori assunti da un attributo CONSIDERANDO SOLO QUANDO IL TARGET E' uguale a 1. Lo faccio prima per i numerical attributes:
![[numclassequal1.png]]
Mi da' la distribuzione dei valori quando y=1 per ogni attributo.  Qui ho molta difficolta' nel capire correlazioni. Quando l'attributo u7 ad esempio ha un valore basso (-6.0) allora ho che spesso la classe predictata e' y=1. Quando l'attributo u7 cresce di valore allora la frazione di punti che hanno y=1 e' molto minore (vedi che ci sono pochi punti? si vede, le barre sono meno alte). Questo mi fa supporre che ci sia CORRELATION perche' all'aumentare di u7 ho che y tende a passare da 1 a 0 (negative correlation no?). Per u1 ho una cosa molto meno linear correlation (ho non linear pattern in cui il valore scende poi sale.).
E per i CATEGORICAL ATTRIBUTES
![[catclassequal1.png]]
Mi da' la distribuzione degli 0 e degli 1 quando y=1.
Questo serve per capire quanto UN VALORE e' CORRELATED ALL'OUTPUT. Ad esempio in u10, il valore di 1 e' MOLTO alto (ci sono un sacco di 1.0 con y= 1). CI sono molti meno 0,0. Questo puo' indicare una correlazione perche significa che quando u10 aumenta (passa da 0 a 1) ho che y aumenta (passa da 0 a 1): ho positive correlation sembrerebbe.
Mi creo quindi NELLA TESTA dei **CANDIDATE CORRELATE** che sono attributi che POTREBBERO ESSERE CORRELATI ma ancora non lo so.

Faccio per tagliare la testa al toro una correlation matrix:
![[correlationmatrix.png]]
Le correlation e' SPARSE e c'e' una **WEAK LINEAR CORRELATION** con la y.
La limitazione della correlation matrix e' questa:
1. se uso la pearson correlation coefficient (come il caso sopra) SI GUARDA SOLO LA LINEAR CORRELATION. Ci potrebbe essere una FORTE CORRELAZIONE TRA ATTRIBUTI E y MA NON LINEAR e io non posso saperlo. 
2. posso usare anche la spearmen correlation, che catcha una monotonic relation tra un valore e un altro. E' una misura di MONOTONIC CORRELATION
3. c'e' anche kendall correlation. E' sempre una monotonic correlation ma c'e' qualcosa di diverso qui.
In tutti i casi ho o LINEAR o MONOTONIC CORRELATIONS, non ho altro (non ho non linear).

**QUAL E' ORA IL NOSTRO GOAL, dopo aver inspectato il DATASET**?
## Use Case Objective
Voglio trovare CORRELATIONS non linear magari, e NON SOLO UNIVARIATE (ma considerando anche piu'' attributi insieme quindi).
Voglio ottenere **CAUSAL RELATIONSHIPS**, voglio trovare **CAUSALITY**. 
**Unlike in classical ML tasks, we don't have an _estimation_ problem**
Rather, our goal is _understanding_ the process behind the data
* We want to identify the true _correlates_ among our candidates
* We want to see _how_ they are linked to the target $y$

**In an ideal world, we'd like to know about _causal_ relationships**
...But in practice, we'll need to be happy with correlations
* Studying causality is indeed possible (a good start is [Judea Pearl's book](https://www.cambridge.org/core/journals/econometric-theory/article/causality-models-reasoning-and-inference-by-judea-pearl-cambridge-university-press-2000/DA2D9ABB0AD3DAC95AE7B3081FCDF139))
* ...But also very challenging, and there's no general and mature tool available
So, we'll count on the domain expert to check the correlations

**Our setup also explains a quirk in the dataset**
All variables except the target are called $U_j$, for "unknown"
* This is synthetic data, so nothing is really unknown
* In fact, the ground truth process linking $Y$ to $U$ is avaialable

**However, for the sake of the lecture, such process will be hidden**
* We will analyze the data pretending we have no such knowledge
* _At the end_ of our exercise we'll check the ground truth
...And we'll see how close we got to the truth!


# Baseline approach
**Our goal is _understading_ the process behind the data**

Of of many possible ways to do it consist in:

* Training an approximate model via Machine Learning
* **Studying the model as a proxy for the real process** (quindi tratto il mio modello come SURROGATO del processo)
**Basically, we use a ML model as an _analysis tool_**
Cioe' faccio le mie conclusioni dal MODELLO ML creato. Studio le PROPRIETA' dell'ML model trainable.
Ho bisogno di qualcosa di INTERPRETABILE, un modellop interpretabile sicuramente.

**For this approach to work, we need the ML model to be _explainable_**
* A few model naturally enjoy this property (e.g. linear models, simple DTs)
* Explaining other models is not obvious (e.g. Neural Networks, large ensembles)
We will start with the simplest option: **Logistic Regression**

## Data Preprocessing
**We start with the usual data preprocessing**
CONSIDERO TUTTI GLI INPUTS COME **CANDIDATE CORRELATES**.
Faccio STANDARDIZATION e divido in train, test sets. Ma perche' ho bisogno di un test set per studiare un modello? BEH no, se devo solo studiare le proprieta' del fitted model no. Ma mi serve il test set per EVITARE CHE IO OVERFITTI. Ovvviamente se il mdoello overfitta NON SARA' UN BUON PROXY. Ho quindi bisogno di un test set per AVERE UN BUON MODELLO DA STUDIARE.
In questo tipo di PROBLEMI OVERFITTING E' LA COSA PEGGIORE CHE POSSO AVERE.

Applico L1 REGULARIZATION (LASSO)
Scikit learn support L1 regularizers for Logistic Regression in the form:
$$
\mathop{\mathrm{argmin}_\theta} H(y, f(x, \theta)) + \frac{1}{C}\|\theta\|_1
$$
* We encourage the weights to be close to 0
* ...And we attempt to sparsify the weights
E' quello usato in lezioni passate? Mi sembra di si (o era L2? ad ogni modo era sempre un penalizer simile)

**NOTA** Ho bisogno di un altro solver (prendo _saga_), perche' IL BASE SOLVER non supporta L1 REGULARIZER 
```python
base_est = LogisticRegression(penalty='l1', solver='saga')
param_grid={'C': 1. / np.linspace(1e-1, 1e4, 100)}
gscv = GridSearchCV(base_est, param_grid=param_grid, scoring='roc_auc')
gscv.fit(X_train, y_train)
lr, lr_params = gscv.best_estimator_, gscv.best_params_
```
Faccio hyperparameter tuning sulla C.
Vedo poi se ho OVERFITTATO guardando la performance sul test set.
```python
lr_score_cv, lr_score_test = gscv.best_score_, roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print(f'AUC score for C={lr_params["C"]:.2f}: {lr_score_cv:.2f} (cross-validation), {lr_score_test:.2f} (test)')
```
*AUC score for C=10.00: 0.64 (cross-validation), 0.60 (test)*
* We use the AUC score, **since this is not a real classification problem** (lo sapevo gia')
* **There's no significant overfitting**
Posso ora guardare i valori dei coefficients:
![[coeff.png]]
guardo sia la magnitude che il sign. Cosi vedo sia se c'e' correlazione e se e' positive o negative. Questo approccio e' in grado di dealare con **MULTIVARIATE CORRELATION** (il modello prende in considerazione PIU' FEATURES insieme (at once)).

**COSA VA MALE IN QUESTO APPROCCIO**? 
1. Le features sono assunte essere INDIPENDENTI (non comunicano tra loro), per quanto l'approccio sia MULTIVARIATE. Il logistic regressor NON E' IN GRADO DI TROVARE **feature interactions**, che era il mio obiettivo principale
2. logistic regression e' praticamente un linear model, quidni NON POSSO TROVARE NON-LINEAR CORRELATIONS.
3. solitamente quando uso LASSO, ho che i pesi O sono alti O sono 0. Qua ho piccoli pesi. strana sta cosa. Qua ho sparse weigghts. E' difficile in questo caso CAPIRE IL RUOLO DELLE FEATURES con valori bassi perche' hanno tutti comparable coefficients. QUA  come faccio a scegliere la soglai di CUT per dire quale saranno gli attributi rilevanti e quelli no
4. Il mio logistic regression performa malissimo: la AUC del ROC e' BASSISSIMA (0.6 poco sopra il RANDOM). MI FIDO DI UN MODELLO DEL GENERE? sta cosa e' terribile e e' il problema piu grande.

POSSO MIGLIORARE LA COSA USANDO UN NONLINEAR MODEL in modo da catturare NONLINEAR RELATIONSHIPS. 
Una NN sarebbe una pessima idea (non e' explainable).
Sto parlando DI TREES broooo che possono IMPARARE nonlinear functions (che non sono continue e hanno dei gaps/dei salti)
Il problema dei NON LINEAR MODELS e' che sono meno interpretabili ovviamente e SONO piu' proni all'overfitting (perche' hanno MOLTA variance).
Uso DECISION TREES perche' mi danno INTERPRETATION.
Se SOSPETTO che non ci siano interazioni TRA LE FEATURES POSSO usare linear regression SULL'OUTPUT DEI CALIBRATION LAYERS (che imparano non linear functions). In sto modo avrei imparato NON LINEARITY ma solo legata alle singole features. Quindi e' UNIVARIATE non-linearity. Usando Linear Regression non tengo in considerazione di non-linear interactions (ma solo di linear interactions).
## Gradient Boosted Trees Model
XGBoosts e' una librearia perfetta per trainare i TREES, perche
1. Posso introdurre diversi tipi di LOSSES
2. Posso introdurre REGULARIZATION, che e' applicabile sulle foglie (sull'output)
3. E' compatibile con scikitlearn, quidni posso usarlo come classe Model.
```python
base_est = xgboost.XGBRegressor(objective='reg:logistic', tree_method='hist', importance_type='total_gain')
param_grid={'max_depth': [2, 3, 4], 'n_estimators': list(range(20, 41, 5)), 'reg_lambda': np.linspace(0, 1000, 6)}
gscv = GridSearchCV(base_est, param_grid=param_grid, scoring='roc_auc')
gscv.fit(X, y)
xbm, xbm_params = gscv.best_estimator_, gscv.best_params_
```
**XGBoost is a library for fast, distributed, training of GBT models**
It has support for _multiple loss functions_
* We are using "reg:logistic", which refers binary cross-entropy
...And for _regularization_ (often missing in tree-based models)
* The "reg_lambda" parameter refers to the weight of an L2 regularization term
* ...Which in GBT is applied [to the leaf labels](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
Ho come risultato
*{'max_depth': 3, 'n_estimators': 40, 'reg_lambda': 200.0}*
Ho dopo la gridsearch che la migliore max_depth e' 3. Questo e' uno shallow tree che magari neanche prende in considerazione tutte le features e in cui non ho tutte le correlations. MA ASPETTA, questa e' la depth di OGNI SINGOLO TREE (sto usando Gradient Boosted ricordo che e' un ensemble). Quindi se li aggrego ho relazioni complicate che possono spiegare le correlations.
**NOTA** pero': se avessi avuto Random Forest 3 sarebbe stato poco. 
Qui traino per RESIDUALS in Gradient Boosting, la' invece ho bisogno di deep trees (nella random forest). Rivedi Gradient Boosting: qui non ho biusogno di DEPTH mi concentro sugli shallow trees e VA BENISSIMO 3.

Ma come funziona la regularization?
* Assuming $T$ is the number of leaves and $w_j$ is the label assigned to each leaf
* ...Then the regularization term is in the form $\sum_{k=1}^T w_j^2$

*ho sul train un AUC di 0.81, su test una di 0.8*
Ho un AUC alto e NON ho overfitting. Finalmente POSSO FIDARMI DEL MODELLO.

So di avere un ENSEMBLE DI PIU' NON LINEAR MODELS. Come interpreto Decision Trees Ensembles???
Se ho un singolo tree BASTA PLOTTARLO, ma in sto caso non ne ho uno solo.

## FEATURE IMPORTANCE
POSSO STUDIARE LE **FEATURE IMPORTANCES**. Si possono calcolare in base a QUANTO benefit ho nello SPLITTARE secondo una determinata FEATURE.
Se faccio avg dei gains per ogni feature, ho poi un VALORE RIFERITO ALLA FEATURE che mi dice quanto questa e' IMPORTANTE.
**Feature importance is typicaly presented as this:**
* For each input $x_j$, we sum the associated _gain_ at training time
* Once training is over, we normalize the scores so that they sum up to 1

![[featureimportance.png]]
Nota che qua c'e' MOLTA differenza tra colonne con alti valori e quelli con valori bassi: queli bassi SONO bassi sul serio qua a differenza del LINEAR REGRESSOR. Ho un modo per FARE CUT cazzo adesso. Ora i PESI SONO VERAMENTE SPARSI.

**Howver, there are _other ways to define importance_**
XGBoost supports 5 different approaches:
* "weight": number of times an attribute is used to split (io pensavo a questo prima)
* "gain": average gain associated to splits over an attribute
* "cover": average number of examples for which an attribute is used to decide
* "total_gain": as above, but replacing the average with a sum (c'e' differenza con l'avg se ci pensi)(e' LA feature importance di sopra ASSURDO era somma allora sopra non avg)
* "total_cover": as above, but replacing the average with a sum
![[gaintotalgaintotal.png]]
weight, total_gain e gain sembrano ASSOMIGLIARSI ma neanche troppo. total_cover e' MOLTO DIVERSA DAL RESTO, cosi' come cover.
Come prendo FEATURES IMPORTANTI in questa situazione? Perche' qua SE CAMBIO I CRITERI ho RISULTATI MOLTO DIVERSI. E' un problema questo. Ovviamente posso fixarne una e fare qualche conclusione a riguardo delle features legate a quel criterio ma non e' esaustivo. Molto difficile

**NOTA** inoltre gain e cover per essere COMPUTATI ho bisogno di UN MODELLO E... un DATASET. Quindi FEATURE IMPORTANCE e' computato attraverso un MODELLO e un **DATASET** (quindi NON E' INDIPENDENTE DAL DATASET. Quelle trovate NON SONO PROPRIETA' DEL MODELLO ma del MODELLO INSIEME AL DATASET). Il modello potrebbe infatti scegliere di splittare SECONDO UN ATTRIBUTO per cause legate AL DATASET (magari quell'attributo NON E' IMPORTANTE ma il DATASET e' tale che SEMBRA importante).

## PERMUTATION IMPORTANCE
Cambio il modo in cui computo FEATURE IMPORTANCE.
Dato un dataset $\{x_i, y_i\}_{i=1}^m$ posso valutare le performance del modello su sto dataset con le performance su un altro dataset in cui prendo i valori della feature _j_ e li **permuto**. Ho quindi che la distribuzione dei suoi valori rimarra' UGUALE (perche' sto solo permutando), ma le CORRELATIONS con gli altri valori degli altri attributi saranno TUTTE SFANCULATE. Vedo quindi QUANTO CAMBIAMENTO HO nelle model performances. Se questo e' piccolo allora l'attributo NON E' COSI' IMPORTANTE, se grande allora L'ATTRIBUTO e' IMPORTANTE (significa che e' CORRELATO con gli altri attributi in maniera forte).

Gli scores sono chiamati **PERMUTATION IMPORTANCES**.
Questo approccio e' applicabile a ogni tipo di modello, NON SOLO A UN DT, ma anche a una NN, a un SVM ecc...
PERMUTATION IMPORTANCES sono ROBUSTI wrt SPURIOUS CORRELATIONS che possono esistere tra dati di un TRAINING SET (correlation non vere diciamo, misleading correlations dovute a overfitting). In sto modo sono molto piu' ROBUSTO riguardo a quanto una feature e' importante.
Devo ripetere la permutazione MOLTE VOLTE, perche con solo una volta rischio che la permutazione puo' creare CORRELAZIONI DOVUTE AL CASO. Poi prendo la MEAN e la STD di queste IMPORTANCES visto che ripeto piu' volte.

In SCIKITLEARN si puo fare con:
```python
r_train = permutation_importance(xbm, X_train, y_train, n_repeats=30, random_state=42)
```
![[permutationimportance.png]]
Ho che u10,u12,u13 SONO IMPORTANTI cazzolo.
Posso anche farlo SUL TEST SET sto permutation_importance e vedere se anche li' ho lo stesso comportamento. 

# Additive Feature Attribution
Tutto buono ma C'e' qualcosa che HO perso ABBANDONANDO LOGISTIC REGRESSION. Cosa?
Con LOGISTIC REGRESSION potevo vedere a livello del SINGOLO TRAINING SAMPLE quanto ogni valore veniva DEVIATO attraverso la moltiplicazione COL SUO RISPETTIVO PESO. Potevo AVERE SPIEGAZIONI sul singolo esempio su quanto questo DEVIAVA dall'average over the population. Lo potevo fare cosi:
$$
\theta^T x - \mathbb{E}_{x^\prime \in P(X)} \left[ \theta^T x^\prime \right]
$$
x: il singolo elemento

**Due to linearity, the formula can be rewritten as:**
$$\begin{align}
\theta^T x - \mathbb{E}_{x^\prime \in P(X)} \left[ \theta^T x^\prime \right] & = \theta^T (x - \mathbb{E}_{x^\prime \in P(X)}[x^\prime]) \\
& = \sum_{j = 1}^n \theta_j (x_{j} - \mathbb{E}_{x_j^\prime \in P(X_j)}[x_j^\prime])
\end{align}$$
Meaning that we can assign a value _to every input attribute_:

* If we know the attribute, the model output moves from the trivial prediction
* ...And the change is given by $\phi_{j}(x) = \theta_j (x_{j} - \mathbb{E}_{x_j^\prime \in P(X_j)}[x_j^\prime])$

We call $\phi_{j}(x)$ the _effect_ of attribute $j$ for the example $x$
HO QUINDI UN VALORE LEGATO A ogni attributo per un particolare sample. Quindi vedo la DEVIATION a livello di OGNI SINGOLO ATTRIBUTO per ogni SPECIFICO EXAMPLE nel dataset.

In sto modo POSSO SAPERE PERCHE' UN DETERMINATO INDIVIDUO (che e' un esempio nel dataset) e' DIVERSO, DEVIA DAL COMPORTAMENTO STANDARD e su QUALE ATTRIBUTO dell'individuo questa DEVIAZIONE AVVIENE.

QUESTA COSA NON LA HO PIU'. **Posso RIAVERLA per un NON-LINEAR MODEL**?
Ho quindi bisogno di UN LOCAL LINEAR EXPLAINER nel mio modello.
Questo LINEAR EXPLAINER e' chiamato un **additive attribution model** e e' fatto cosi':
$$
g(z, x) = \phi_0 + \sum_{j = 1}^n \phi_{j}(x) z_j \quad\text{with: } z_{j} \in \{0, 1\}
$$
$z_j$ e' chiamato *simplified input* che indica se il _j-th_ attribute si conosce o no.
Praticamente costruisco un LINEAR EXPLAINER che localmente in base all'input mi dice l'IMPATTO di ogni FEATURE. In sto modo e' LA STESSA COSA DI AVERE LOGISTIC REGRESSION, il che e' buono.
## Shapely Values

**How do we build the additive attribution model?**

* We've already seen how to do it for linear models
* ...But for non-linear models the input features _interact_ with each other
**A possible solution: _marginalizing_ over all subset of remaining features**
Let $\mathcal{X}$ be the set of all input features; then we have:
$$
\phi_{j}(x) = \sum_{\mathcal{S} \subset \mathcal{X} \setminus j}
    \frac{|\mathcal{S}|! (n - |\mathcal{S}| - 1)!}{n!} (\hat{f}(x_{\mathcal{S} \cup j}) - \hat{f}(x_{\mathcal{S}}))
$$
* The sum is over all subsets that do not contain feature $j$
* The coefficient ensures normalization
* $\hat{f}(x_{\mathcal{S}})$ is the model evaluate with only features in $\mathcal{S}$

**The result of our marginalization:**

$$
\phi_{j}(x) = \sum_{\mathcal{S} \subset \mathcal{X} \setminus j}
    \frac{|\mathcal{S}|! (n - |\mathcal{S}| - 1)!}{n!} (\hat{f}(x_{\mathcal{S} \cup j}) - \hat{f}(x_{\mathcal{S}}))
$$

...Are known as _Shapely values_

* They originate from game theory
* ...In a setup where we want to assign credit to multiple actors for a result
* The actors correspond to our input features, the result to the model output
CI SONO DUE PROBLEMI:
1. computare tutti sti subsets e' expensive
2. come faccio per i MISSING ATTRIBUTES? dovrei queryare il modello CON ALCUNI FEATURE VALUES MANCANTI. Non posso sempre farlo. Tipo in NN non si puo'.

## Kernel SHAP

**Those issues can be sidestepped by learning a _local linear approximator_**

Given an example $x$, we can:

* Sample multiple simplified vectors $z^\prime$ of simplified inputs $z$ from $\{0, 1\}^{n}$ (QUESTO VIENE FATTO INVECE DI LOOPPARE OVER TUTTE LE POSSIBILI COMBINAZIONI DI INPUT FEATURES, FACCIO m SAMPLES immagino )
* For every sampled vector, we construce an example:
	  - Forall $j$ s.t. $z_j^\prime = 1$, we put $x^\prime_j = x_j$ in the example
	  - We sample all $x^\prime$ s.t. $z_j^\prime = 0$ from a _background set_ (non chiaro, dice che prende un sample a random qua dietro. Cioe' che tipo quando ho $z_j^\prime = 0$ allora sull'INPUT CHE STO CREANDO prende un valore preso a RANDOM da un background dataset (perche' non posso mettere ovviamente 0 in una NN, devo mettere un valore)). Tutti sti samples $x^\prime$ SERVONO PER FARE MARGINALIZATION di base (servono per stimare l'expectation )
* We train a particular type of linear model on the obtained examples
* ...Then we compute the Shapely values using the linear formula
NON CI HO CAPITO UN CAZZO DIFFICILISSIMO.
Codice:
```python
f = lambda x: xbm.predict(x)
explainer = shap.KernelExplainer(f, shap.sample(X_train, 100), link='logit')
shap_values = explainer(X_test)
with open(os.path.join('..', 'data', 'shap_values.pickle'), 'wb') as fp:
    pickle.dump(shap_values, fp)
```
link='logit' dice COSA voglio spiegare. Quindi la logit, NON la probability.
shap.sample(Xtrain,100) E' il *background set* di sopra.
![[waterfall.png]]
E' riferito AL PRIMO ESEMPIO nel testset.
f(x) rappresenta l'output del modello.
Le bars sono i MIEI SHAPELY VALUES:
1. u13 mi PORTA nella DIREZIONE NEGATIVA, quindi riduce la probabilita' che la classe sia 1
2. u4 mi PORTA nella DIREZIONE POSITIVA quindi AUMENTA la probabilita' che la classe sia 1
3. u5 causa una RIDUZIONE della probabilita'
4. u10 un'AUMENTO
5. tutte le altre sommate causano un AUMENTO di 0.01
![[forceplot.png]]
questo e' il force plot e mi dice le FORCES che insieme SPINGONO il risultato in una determinata direzione che e'  l'output della rete. E' un waterfall comrpessato. Nota che visto che qua ho 0.12 allora la SIGMOID mappera' questa cosa a piu' di 0.5. Praticametne IN QUESTO CASO la PROBABILITA' di avere la condizione y=1 e' LEGGERMENTE SUPERIORE ALLA MEDIA

Poi posso fare GLOBAL FORCE PLOT, in cui per ogni esempio HO un FORCE PLOT sulle COLONNE. Insieme vedo per ogni esempio la speigazione.


![[globalforceplot.png]]
PASSO DA SAMPLES con la maggior parte degli attributi che hanno un POSITIVE EFFECT, a SAMPLES man mano CON LA MAGGIOR PARTE DEGLI ATTRIBUTI che hanno invece NEGATIVE EFFECTS.
Con sto tipo di plots posso anche vedere CLUSTERS di dati: dati che si assomigliano.\
![[u13.png]]
Sto plot invece fa vedere l'effetto di u13. Praticamente quando si hanno input values per u13 che sono ALTI allora gli shapy values saranno ALTI (quidni PUSHERANNO il risultato verso una ZONA POSITIVA, che quindi predicti la presenza della condizione y=1). 
POI HO IL BEESWARM PLOT
![[bees.png]]
E' molto informativo.
Praticamente ho in ogni riga un ATTRIBUTO. Per ogni attributo mostro con un punto quando QUESTO e' legato a un particolare SHAPY VALUE. Nota che nel caso di u13 ho il COMPORTAMENTO osservato SOPRA in cui i valori degli shapy values AUMENTANO. Praticamente BRO il MAP e' quello che vedi. Il colore e' legato ALLA PREDICTION FINALE (che non dipende SOLTANTO DALLO SHAPY VALUE legato a quella feature ovviamente). Quindi se metti che nonostante lo shapyvalue sale epr una determinata feature ma il COLORE e' comunque BLUE significa che CI SONO INTERAZIONI CHE (nonostante l'attuale feature spinge per predictare il valore POSITIVO) portano l'outcome finale AL NEGATIVO (valore blue) (dipende infatti dal valore degli shapy values che altre features prendono).
![[u12u4.png]]
In sto plot VEDO SE CI SONO POSSIBILI INTERAZIONI TRA FEATURES. Quindi prendo i shapy values di u12 al CAMBIARE, all'aumentare dei valori prendibili dall'attributo u12 e do a ogni punto UN COLORE in base a QUANTO E' ALTO LO SHAPY VALUE di QUEL PUNTO su u4. NOTO che non c''e grossa INTERAZIONE (quando lo shapy value di uno aumenta l'altro NON SEMPRE AUMENTA o NON SEMPRE diminiusce insomma). Credo che si avrebbe INTERAZIONE se a livello di quello shapy value in u12 corrisponda uno shapy value dello stesso valore anche in u4. 

**NOTA** : problemi:
1. non ho ancora un modo per capire le RELEVANT FEATURES
2. Ho permutation importance PER una global evaluation e 
	SHAP per local evaluation delle features
NON HO LO STESSO APPROCCIO PER LOCAL E GLOBAL
Uso per risolvere il punto 2. SHAP per computare FEATURE IMPORTANCES.
**SHAP explanations can be aggreated to get global importance scores**

By default, this is done by averaring absolute SHAP values:

$$
\bar{\phi}_j(x) = \frac{1}{n} \sum_{i=1}^m |\phi_j(x_i)|
$$

* Other aggregation functions can also be used (e.g. max)
**The SHAP library provide convenience functions to plot aggregated values**

Here's how to plot mean (absolute) SHAP values:
![[featureimportancewiashape.png]]
Come fixo il problema 1.?
**A viable approach for feature selection consists in solving:**

$$
\mathop{\mathrm{argmin}}_{\mathcal{S} \subseteq \mathcal{X}} \left\{ |\mathcal{S}| : \hat{y} = \hat{f}_{\mathcal{S}}\left(x_{\mathcal{S}}\right), L\left(y, \hat{y}\right) \leq \theta \right\}
$$

Where $x, y$ denote all the training data. Intuitively:

* We search for the smallest subset of features $\mathcal{S}$
* ...Such that a model $\hat{f}_{\mathcal{S}}$ trained over only over them
* ...Still has an acceptable (cross-validation) accuracy
Heuristics (e.g. greedy search) can be used to improve scalability
per avere $\hat{f}_{\mathcal{S}}$ nota devo fare RETRAINING.


**This optimization-driven approach**

* ...Can be customized by adjusting the constraint and cost function
* ...Can reduce data storage and location costs on the deployed model

**NOTA** nel nost4ro caso QUESTO APPROCCIO NON VA BENE.
1. Non ho una threshold se ci pensi. 
2. Non ho neanche UN COST per la LOSS OF ACCURACY.
3. Inoltre NON VOGLIO TROVARE IL MINIMO SET OF FEATURES, ma le FEATURES CHE CORRELATES tra loro.
Voglio un different approach.
Nelle slides dice:
**For a number of reasons:**
* We care about finding _all the relevant features_, not a minimal set
* How should the accuracy threshold be calibrated?
* What about the noise induced by retraining?
## STATISTICAL HYPOTHESIS TESTING
Usero' questo bro.
Parto praticamente da UN'IPOTESI H. INvece di dire che e' vera e dimostrarlo COLLEZIONO PIU' EVIDENZE POSSIBILI per dire che H e' vero.
Inizio formulando una NULL HYPOTESIS che dice "H non e' vera" che e' una COMPETING HYPOTHESIS.
Il nostro obiettivo diventa ora di RIFIUTARE la NULL HYPOTESIS. 
Se riesco a rifiutarla allora L'IPOTESI H INIZIALE la considero VERA.
COME RIFIUTO LA NULL HYPO:
1. devo definire un TEST STATISTIC T(X) che sono statistiche su esperimenti. Voglio che questo sia MONOTONICALLY RELATED TO H (tipo che se T e' grande allora $H_0$ (la null ipotesis) e' probabilmente false, se e' piccolo allora la null e' vera).
2. Defninisco la PROBABILITA'; TEORICA di T(X) rispetto alla NULL HYPOTHESIS qunidi P(T(X)| $H_0$)
3. La calcolo POI EMPIRICA SUI MIEI DATI t = P(T(x)| $H_0$)
4. Calcolo poi la probabilita' che T(X) sia almeno grande quanto t data la NULL HYPOTESIS $$
p = P(T(X) \geq t \mid H_0)
$$
5. If $p \leq 1 - \alpha$ for some confidence $\alpha$, we _reject the null hypothesis_
	**NOTA** Se la PROBABILITA' p E' alta ALLORA NON POSSO RIGETTARE LA NULL HYPOTESIS, perche' significherebbe che QUELLO CHE STO OSSERVANDO E' LIKELY under the null hypotesis (e' probabile). E io invece voglio che sia improbabile per rigettarla.

**First, we need to define our _hypothesis_ and _data_**
We care about identifying correlates, so a possible choice might be:
* $H \equiv \text{"}r(X, Y) \geq r^*\text{"}$, for some correlation measure $r$ (dico che questa cosa per me e' vera)
* $\text{data} \equiv \text{"}\text{x, y}\text{"}$, i.e. our sample (un po' di osservazioni)
La NULL HYPOTESIS $H_0$ sara'  $H_0 \equiv \text{"the observed result is due to chance"}$ (prima infatti in H abbiamo assunto che il result e' dovuto a qualche tipo di  correlazione)
* In most cases, the null hypothesis assumes what we observe is due to chance
* If we manage to reject it, we can claim that $H$ is more likely true
* The tricky part is choosing a $H_0$ for which we can compute probabilities
* ...Without introducing _unnecessary assumptions_

Now we need some "test" related to $H$ and $H_0$<br>
...And it must be something for which we can compute probabilities
How do we do that?
DEVO ORA RUNNARE DEGLI ESPERIMENTI CHE MI STIMINO se il valore del correlation coefficient sia SOPRA O SOTTO $r^*$.

**Let's consider the event $\text{"}r(X, Y) < r^*\text{"}$**
Since it has a binary outcome, it will follow a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
* If we assume that the correlation is due to chance...
* ...Then the associated probability should be $^1/_2$

**Let's pretend we make repeated experiments**
The _number of observed events $T(X, Y)$_ will follow a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
T(X,Y) E' LA TEST STATISTICA SUI MIEI DATI.
* Given the number of experiments $n$
* ...The probability of $T(X, Y)$ should be $P(T \mid H_0) = B(n, {^1/_2})$
* P(T|H_0) e' la PROBABILITA' TEORIETICA
DEVO ORA STIMARE LA PROBABILITA' EMPIRICA di sta cosa usando i miei dati.
Devo SIMULARE IL FATTO CHE LE CORRELAZIONI SIANO RANDOMICHE (HO H_0 come qualcosa di DATO). Voglio AGGIUNGERE RANDOMNESS senza cambiare la DISTRIBUTION
Posso BREAKKARE LE CORRELATIONS ON PURPOUSE  a livello di colonne cmome ho fatto in permutation IMPORTANCE come si chiama.
Posso shufflare a livello di COLONNE. RIPETO IL PROCESSO PIU' VOLTE

**NOTA** il null hypo NON e' l'opposto di H e' un Competitore.

**manca una grande parte**
STE COSE VISTE OGGI SONO IMPORTANTI MANNAGGIA LA MADONNA, SONO PROPRIO RIFERITE ALL'INTERPRETABILITA'.