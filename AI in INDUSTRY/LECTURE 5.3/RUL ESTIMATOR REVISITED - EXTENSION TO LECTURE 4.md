**The RUL estimation was of very poor quality**
* Our model was good enough for defining a policy
* ...But not usable to provide a real-time RUL estimate

**Why did we fail? Here are a few potential culprits**
* Are we sure our target is correct? What if the defect arises late?
* Our target looks deterministic: are we accounting for uncertainty?
* Are we providing all the necessary input?

# RETHINKING
* We start by defining a _probabilistic model_
* We use _ML to approximate_ key components of such model
* We use the model + the approximators to make _probabilistic predictions_

**We are interested in the "survival time" of an entity**
We can start by modeling that as a single random variable $T$ with unknown distribution
$$
T \sim P(T) \qquad \text{(draft 1)}
$$
* $T$ (with $\mathbb{R}^+$ as support) represents the survival time (dai sarebbe il mio target no?)

**To be specific, we want $T$ to be _remaining_ survival time**
...With respect to time $t$ when we perform the estimation. Formally:
$$
T \sim P(T \mid t)  \qquad \text{(draft 2)}
$$
* Now the distribution is conditioned on $t$ (which we can access)
**Survival depends on additional factors**
E.g. on the lifestyle of a person, or on how industrial equipment is used
* We can model these factors as additional random variables
* We can distinguish between behavior in the past $X_{\leq t}$ and the future $X_{>t}$
**Formally, we have:**
$$
T \sim P(T \mid X_{\leq t}, t, X_{> t})  \qquad \text{(draft 3)}
$$
For now we focus on _capturing the elements that affect the estimate_
* We not not care (yet) about the fact that we can access them
* The idea is to focus on _one problem at a time_
**...But of course whether a quantity can be accessed or not does matter**
In particular, _future behavior cannot be accessed_ at estimation time (EH che ne so io quando faccio prediction e uso il modello in PRODUZIONE cosa succedera' DOPO lol)
* Intuitively, future behavior affects the estimate as _noise_
* Formally, we can _average out_ its effect
**This operation is called _marginalization_ and leads to:**
$$
T \sim \mathbb{E}_{X_{> t}} \left[ P(T \mid X_{\leq t}, t, X_{> t}) \right]  \qquad \text{(draft 4)}
$$
This is a good model for the distribution of the variable we wish to estimate
* The "sawtooth like" target that we used earlier for RUL regression
* ....Corresponds to _samples from $P(T \mid X_{\leq t}, t, X_{> t})$_

## Look at our previous model
**In the RUL lecture we trained a regressor**
...With the current parameters/sensors as input and an MSE loss
* Meaning the _our estimator_ is making implicitly use of this model:
$$
T \sim \mathcal{N}(\mu(X_{t}), \sigma)
$$
* $\mathcal{N}$ denotes the Normal distribution, $\mu(\cdot)$ represents our old regressor (cioe' praticamente avevo un modello che si faceva predizioni deterministiche, e questa cosa e' rappresentata da $\mu(.)$). Ovviamente c'e' un NOISE nella predizione quindi dico che _T_ e' samplata da una Gaussian centrata sulla mia predizione fatta col vecchio modello  $\mu(\cdot)$ e con std pari a $\sigma$ che NOTA e' UGUALE PER OGNI PUNTO $X_t$). Quindi ho error noise che E' UGUALE per punti VICINO ALLA FINE DELLA MACCHINA (quando la macchina si sta per rompere) e LONTANO (molti timesteps lontano), quando invece L'INCERTEZZA DOVREBBE ESSERE ALTISSIMA.

**Now, compare it with our "ideal" probabilistic model:**
$$
T \sim \mathbb{E}_{X_{> t}} \left[ P(T \mid X_{\leq t}, t, X_{> t}) \right]
$$
* Let's try to spot together any major difference
### Differenze col corrente modello
**Now, compare it with our "ideal" probabilistic model:**

$$
T \sim \mathbb{E}_{X_{> t}} \left[ P(T \mid X_{\leq t}, t, X_{> t}) \right]
$$

Let's try to spot together any major difference
1. We considered _a single $X_{t}$_, rather than $X_{\leq t}$ (Actually, we tried that by using sequence input, it helped, but not much)
2. We _disregarded time_ as an input
3. We assumed a _Normal distribution_ with _fixed variance_
	* It's unclear how to relax the normality assumption
	* ...But we know we can fix the variance this using a neuro-probabilistic model! (e' quello che praticamente voglio fare. Avere una STD sulla prediction che sia diversa in base all'INPUT che gli mostro. Quindi magari piu' grande se sono incerto, meno grande se sono certo(e questo lo capisce la NN dall'input)).
**CRUCIALE CHE TU LO COMPRENDA**: il neuroprobabilistic model mi permette DI DARE IN OUTPUT UNA PREDIZIONE DOTATA DI MEAN E STANDARD DEVIATION. STA STANDARD DEVIATION VIENE IMPARATA, COSI' COME LA MEAN. Quindi serve a questo di base.

**We make our ML model capable of _estimating variance_**
In particular, we can use a neuro-probabilistic ML model
* The underlying probabilistic model is:
$$
T \sim \mathcal{N}(\mu(X_t, t), \sigma(X_t, t))
$$
In practice:
* We use conventional ML model (a network) to estimate $\mu$ and $\sigma$
* ...Then we feed both parameters to a `DistributionLambda` layer\
Nota come uso solo $X_t$ : intuitivamente sarebbe stato meglio l'approccio con le sliding windows per fare PROXY di $X_{\leq t}$  MA abbiamo gia' visto nella LECTURE 4 che UN SOLO INPUT BASTAVA PER CATTURARE TUTTA LA LOCAL CORRELATION (quindi markov property gia' rispettata con SOLO UN INPUT).
**Our model will be able to learn how $\sigma$ depends on the input**
* This will be more challenging, but also more flexible
* ...And it will provide us confidence intervals

# Building the Neuroprobabilistic model
****Code to build the model can found in the `util` module**

```python
def build_nn_normal_model(input_shape, hidden, stddev_guess=1):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for h in hidden:
        x = layers.Dense(h, activation='relu')(x)
    mu_logsigma = layers.Dense(2, activation='linear')(x)
    lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], scale=tf.math.exp(t[:, 1:]))
    model_out = tfp.layers.DistributionLambda(lf)(mu_logsigma)
    model = keras.Model(model_in, model_out)
    return model
```

* Note the way the input tensor `t` is split in the `lambda` function
* That is needed to obtain the correct tensor shapes (columns)**

```python
tr_rul_std = tr_s['rul'].std()
nnp = util.build_nn_normal_model(input_shape=(len(dt_in), ), hidden=[32], stddev_guess=tr_rul_std)
util.plot_nn_model(nnp)

tr_rul_std = tr_s['rul'].std()
nnp = util.build_nn_normal_model(input_shape=(len(dt_in), ), hidden=[32], stddev_guess=tr_rul_std)
util.plot_nn_model(nnp)
```

```python
negloglikelihood = lambda y_true, dist: -dist.log_prob(y_true)
nnp = util.build_nn_normal_model(input_shape=(len(dt_in), ), hidden=[32], stddev_guess=tr_rul_std)
history = util.train_nn_model(nnp, tr_s[dt_in], tr_s['rul'], loss=negloglikelihood, epochs=70, verbose=0, patience=10, batch_size=32, validation_split=0.0)
util.plot_training_history(history, figsize=figsize)
```

# Evaluation
Mi interessa la distribuzione STIMATA non **SAMPLARE**. Quindi invce di usare _predict_ chiamo DIRETTAMENTE il modello.
```
nn_pred_ts = nnp(tr_s[dt_in].values)
```
Da questi distribution objects posso prendere **means e standard deviations**
```python
np_pred_ts_mean = nn_pred_ts.mean().numpy().ravel() * trmaxrul
np_pred_ts_std = nn_pred_ts.stddev().numpy().ravel() * trmaxrul
```

![[RULneuroprobabilistic.png]]
* The initial plateaus in the predictions have disappeared
* ...And the true RUL is typically within 1$\sigma$ from the predicted mean

**The approach we have seen works already very well**
* We get a predicted mean (as usual)
* ...But also an input-dependent standard deviation
MA NON POSSIAMO FARE LA STESSA COSA CON SAMPLE WEIGHTS?
Cioe' non e' uguale?
**Yes, but it's not the same**
* Sample weights allow use to control the standard deviation with an MSE loss
* ...But we need to _pre-compute them_ using another model (o li computiamo STATICI prima)
**They cannot be learned in an end-to-end fashion!**

# Survival Analysis via Neural Models
**Our probabilistic RUL model worked quite well**
...But it still has some weak spots
1. Sto usando un'ASSUNZIONE sulla DISTRIBUZIONE della predizione (ovvero una Normal Distribution). Ma non so se e' la migliore assunzione tra le possibili distribuzioni esistenti. :c
2. La nostra predizione dipende dal COMPORTAMENTO FUTURO  (ricorda la MARGINALIZATION dell'Expectation). Il modello COMPUTA QUESTA Expectation on FUTURE BEHAVIOUR attraverso il TRAINING SET.  Perche' ci sono PUNTI SIMILI TRA LORO che PORTANO a degli OUTCOME completamente diversi tra loro. Questo mi MARGINALIZZA il future behavious (questa e' una cosa che fanno tutti i modelli di machine learning). Se pero' ho un training set poco rappresentativo, ho che da un INPUT vengo portato a MOLTI MENO OUTCOME magari, e quando vedo quell'input nel test set che magari PORTA A UN ALTRO OUTCOME che non era presente nel TRAINING SET allora li' il modello mi performa una merda. QUINDI QUAL'E' IL PROBLEMA COL MIO MODELLO PRECEDENTE: che segue un buon approccio, ma MARGINALIZZARE i FUTURE OUTCOMES e' DATA HUNGRY (e mi sa che non bastano quelli che ho per fare JACKPOT)(questo e' un grandissimo problema del RUL).

## Censoring
**In many domains, run-to-failure experiments are expensive to obtain**
...But _partial runs_ might abunant
* Broken industrial machines vs regularly maintained ones
* Deaths in organ transplant waiting lists vs alive patients
Il concetto e' che e' molto piu' semplice avere dati di COSE CHE FUNZIONANO piuttosto che dati di esperimenti RUN-TO-FAILURE (pensa agli UMANI, e' piu' semplice ottenere dati (anche piu' etico) di una persona mentre e' in vita, piuttosto che di una persona fino a che non muore).
**The C-MAPSS dataset is very unrealistic from this point of view**
The simulator is good, but there are way too many experiments
* We can simulate limited availability of supervised data
* ...By randomly truncating a portion of the training set
Faccio Censoring quindi, per creare un environment (praticamente semi-supervised) PIU' REALISTICO, in cui ho punti con labels e punti senza labels.
**CENSORING** sarebbe quando TRONCO una SERIE randomicamente, in sto caso LA RUL.
QUESTO MI PORTA A AVERE LA RUL DEGLI ESPERIMENTI TRONCATA COME IN QUESTO CASO:
![[truncatedRUL.png]]
Quando la linea si  interrompe significa che e' stato troncato l'esperimento, e non si sa quanto POTEVA ANCORA ANDARE GIU' (pero' aspetta, io IN TEORIA CE L'HO IL VALORE DI RUL con cui poter dire quanto manca alla macchina da vivere no? Se tronco a 160 di RUL so che mancano 160 steps no? IMMAGINO CHE IN REALTA' NON POSSO USARE STA INFO. Eh no che non puoi danie' e' proprio quello il punto).
**NOTA*** 
* We still can plot the RUL values, but only since we used _simulated_ censoring
* In a real use case, we would have _no RUL target for this data_

comunque...
Posso usare un HEALTHY SIGNAL con l'UNSUPERVISED DATA.
Vabe ad ogni modo faccio cosi:

**We could study the distribution of $T$ via its _survival function_**
The survival function of a variable $T$ is defined as:
$$
S(t) = P(T > t)
$$
I.e. it the probability that the entity "survives" at least until time $t$
* It is the complement of the cumulative probability function $F(t) = P(T \leq t)$
NOTA che T rappresenta da QUANTO TEMPO UNA MACCHINA E' VIVA, da quanto tempo funziona.

Posso anche considerare LO STATO DELLA MACCHINA, quindi la SURVIVAL FUNCTION DIVENTA:
$$
S(t, X_{\leq t}) = P(T > t \mid X_{\leq t})
$$
Mi ritorna in sto modo la probabilita' che la macchina CAMPI piu' di un timestep t data la configurazione della macchina nei timesteps prima di t.
Per come e' definita, NON puo' Prendere in considerazione IL FUTURO. quindi e' completamente definita basandosi sul PASSATO (questo e' un downside). Pero' dall'altra parte NON OVERFITTO. In pratica NON PERDO NIENTE (perche' anche nei modelli precedenti non prendevo in considerazione il futuro) e GUADAGNO qualcosa (ovvero ilfatto che non overfittero' perche' l'introduzione dei CENSORING e la formulazione del survival function me lo impediscono- NON HO MARGINALIZATION PRATICAMENTE PERCHE' HO CENSORING).

Se il tempo e' discreto  (e lo e' nel nostro caso) (ho operating cycles) allora posso FATTORIZZARE LA SURVIVAL FUNCTION cosi:


$$
S(t, X_{\leq t}) = (1 - \lambda(t, X_{t})) (1 - \lambda(t-1, X_{t-1})) \ldots
$$

Where $\lambda$ is called _hazard function_
**The hazard function is _a conditional probability_**
...That of not surving one more step. Formally:
* $\lambda(t, X_{t})$ is the probability of _not surviving_  at time $t$
* ...Given that the entity _has survived_ until time $t-1$. I.e.:
$$
\lambda(t, X_{t}) = P(T = t \mid T > t-1, X_{t})
$$
E' praticamente la probabilita' di NON SOPRAVVIVERE  a timestep t (indicata dal fatto che la random variable ASSUME proprio il valore t) dato il fatto che la T e' sopravvissuta fino a timestep t-1 e data l'osservazione corrente X_t.
As a side effect, $\lambda$ only depends on _one_ observation (nota che e' CONDITIONAL e dipende solo dall'entry corrente $X_t$, NON da tutto il passato).
Quindi _S_ e' il prodotto delle probabilita' di sopravvivere a ogni step, dall'inizio fino allo step _t_.

## Our Plan

**We will attempt to _train an estimator $\hat{\lambda}_\theta(t, x_t)$_ for the hazard function**
Se riesco a stimarla allora POTRO' STIMARE ANCHE LA SURVIVAL FUNCTION.
NOTA CHE posso ottimizzare questa hazard function ANCHE in scenari in cui ho **CENSORING**, anzi forse e' una delle poche soluzioni che posso adottare: se c'e' un troncamento in un esperimento, posso comunque dire che la macchina e' sopravvissuta di UNO step per ogni STEP presente (anche se poi non si e' arrivati effettivamente alla morte della macchina CAUSA CENSORING).
LA COSA NEGATIVA (come gia' detto ormai) e' che non posso dire nulla sul FUTURO.

**Additionally, $S$ and $\lambda$ have more limited uses**
We can still _define a threshold-based policy_, e.g. by checking whether:
$$
\hat{\lambda}_\theta(t, x_t) \geq \varepsilon
$$
...But we'll see that _making forecasts_ is not trivial and requires approximations
Quindi se uno di sti $\lambda$ ha il valore maggiore uguale alla $\epsilon$, allora quello e' il timestep in cui devo chiamare MANTAINANCE.

## Training a Hazard Estimator

**Before we get that, we need a way to train our $\hat{\lambda}_\theta$ estimator**

We can start by modeling the _probability of a survival event_

* Say the $k$-th experiment in our dataset ends at time $e_k$
* Then the corresponding probability according to our estimator is:

$$
\hat{\lambda}_\theta(e_k, x_{k,e_k}) \prod_{t = 1}^{e_k-1} (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$

Where $x_{k,t}$ is the available input data for experiment $k$ at time $t$

**This is the probability of:**

* Surviving all time steps from $1$ to $e_k-1$
* Not surviving at time $e_k$

**We can now formulate a likelihood maximization problem**

Assuming we have $m$ experiments, we get:

$$
\mathop{\text{argmax}}_{\theta} \prod_{k=1}^{m} \hat{\lambda}_\theta(e_k, x_{k,e_k}) \prod_{t = 1}^{e_k-1} (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$


**Then, let's rewrite the formula:**

* Let $d_{kt} = 1$ iff $t = e_k$, i.e. if the experiment ends at time $k$
* ...And let $d_{kt} = 0$ otherwise. Then we can get:

$$
\mathop{\text{argmax}}_{\theta} \prod_{k=1}^{m} \prod_{t = 1}^{e_k} d_{k,t} \hat{\lambda}_\theta(t, x_{k,t}) + (1 - d_{k,t}) (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$

Now the two products can be freely swapped

**Starting from:**

$$
\mathop{\text{argmax}}_{\theta} \prod_{k=1}^{m} \prod_{t = 1}^{e_k} d_{k,t} \hat{\lambda}_\theta(t, x_{k,t}) + (1 - d_{k,t}) (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$

We obtain an equivalent problem through a log transformation:

$$
\mathop{\text{argmax}_\theta} \sum_{k=1}^{m} \sum_{t = 1}^{e_k} \log \left( d_{k,t} \hat{\lambda}_\theta(t, x_{k,t}) + (1 - d_{k,t}) (1 - \hat{\lambda}_\theta(t, x_{k,t})) \right)
$$

Since either $d_{k,t} = 1$ or $d_{k,t} = 0$, we can also split the log argument:

$$
\mathop{\text{argmax}_\theta} \sum_{k=1}^{m} \sum_{t = 1}^{e_k} d_{k,t} \log \hat{\lambda}_\theta(t, x_{k,t}) + (1 - d_{k,t}) \log (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$
**Finally, with a sign switch we get:**

$$
\mathop{\text{argmin}_\theta} - \sum_{k=1}^{m} \sum_{t = 1}^{e_k} d_{k,t} \log \hat{\lambda}_\theta(t, x_{k,t}) + (1 - d_{k,t}) \log (1 - \hat{\lambda}_\theta(t, x_{k,t}))
$$

> **Does this remind you of something?** 
> BINARY CROSS ENTROPY BRODERRR\

**This is a (binary) _crossentropy minimization_ problem!**

* $d_{k,t}$ has the same role as a class
* $\hat{\lambda}_\theta(t, x_{k,t})$ is the model output
* We have a sample for every experiment and time step (the double summation)

**This means that our $\hat{\lambda}_\theta$ can be seen _as a classifier_**

* We just need to consider all samples in our dataset individually
* Then attach to them a class corresponding to $d_{kt}$
* ...And finally we can train a neural classifier as usual

The model output will be _an estimate of the hazard function_
**This is almost precisely _what we did in our classification approach_**

...But now we have _a much better interpretation_

* We know how to define the classes
* We better know how to interpret the output
* We know the semantic for a threshold-based policy
* We know that we can safely deal with censoring

NOTA che io DICO che la macchina MUORE (quindi associo la label 1 a un determinato sample) QUANDO praticamente l'esperimento della macchina e' FINITO (che nel caso di Censoring coincide con FINO A CHE NON ho piu' dati disponibili per quell'esperimento). Pero' e' l'unico modo che ho, e ci sta.


## Effect on Censoring on the Distribution

**The new approach allows us to use censored data**

This is good, but it also has the effect of altering the distribution

* For end-to-failure experiments, are samples follow their natural distribution
* ...But censored data includes no end event, causing a skew
Ho praticamente la distribuzione dei segnali (o 0 o 1) che e' diversa da quella che avrei avuto se non avessi avuto il Censoring. (Cioe' io metto 1 magari quando alla macchina mancano ancora molti steps di vita, solo perche' NON ho i dati per la sua rimanente vita causa CENSORING).

Faccio quindi **IMPORTANCE SAMPLING**.(non chiarissimo come faccia rivedilo)
Traino e poi GUARDO ALLE MIE PREDIZIONI care:
![[estimatedhazard.png]]
Ho che la PROBABILITA' di NON sopravvivere un altro step e' molto bassa. Poi ALLA FINE AUMENTA MOLTO.

POSSO STOPPARE LA MACCHINA QUANDO L'estimated hazard diventa TROPPO grande:
**We can define a policy based on the $\hat{\lambda}_\theta$ estimator as usual**
Namely, we trigger maintenance when:

$$
\hat{\lambda}_\theta(t, x_t) \geq \varepsilon
$$

The threshold can be defined again based on some cost metric

## FORECASTING
POSSO fare una cosa fighissima con questo modello, ovvero fare FORECASTING (predizioni nel futuro).
**We can use $\hat{\lambda}_\theta$ to perform forecasting**

In particular, we know the probability of surving _$n$ more_ steps is given by:

$$
\frac{S(t+n)}{S(t)} = \prod_{h=0}^{n} (1 - \lambda(t+h, X_{t+h}))
$$

...Which we can approximate (for a run $k$) as:

$$
\frac{S(t+n)}{S(t)} \simeq \prod_{h=0}^{n} (1 - \hat{\lambda}_\theta(t+h, x_{k,t+h}))
$$

* In theory, we can forecast survival probabilities arbitrarily far
* ...But in practice there is an issue

**The formula requires access to _future values_ of the $X_t$ variable**

$$
\frac{S(t+n)}{S(t)} \simeq \prod_{h=0}^{n} (1 - \hat{\lambda}_\theta(t+h, \color{red}{x_{k,t+h}}))
$$

* Unfortunately, we cannot access those in real life :-(
* We have two main options to deal with this
(Ho punti FUTURI che NON POSSO SAPERE bro. Tra i CONDITIONING FACTORS cho anche gli $x_{k, t+h}$).


**First, can _ignore time-varying input_ in our estimator**
Formally, this is the same as marginalizing out all time-varying factors
* $\hat{\lambda}_\theta(t, x_t)$ becomes $\hat{\lambda}_\theta(t, x)$, for a fixed $x$
* $x$ represents some stable information, e.g. component type, genetics
In some cases, this is perfectly viable approach

**Second, we can attempt to _predict future $x_t$_ values**
This is viable as long as our predictions are good enough
* We can use a second ML estimator to predict $x_t$
* ...Or as a special case we can rely on the simple _persistence model_

**In practice, we just assume $x_t$ is stable for some time**
With this simple assumption, we get:
$$
\frac{S(t+n)}{S(t)} \simeq \prod_{h=0}^{n} (1 - \hat{\lambda}_\theta(t+h, x_{k,t}))
$$
* Unlike the original expression, this is easy to compute
* ...And it might be a reasonable approximation for shorter time horizons
QUINDI ASSUMO CHE X rimane LO STESSO HAHAHAHA, eppure funziona eh. Cambio quindi solo la t.
Facciamolo per 300 steps:
```python
ref_sample = tr_s.iloc[220]
look_ahead = 300
hazard = util.predict_cf(nnl, ref_sample[dt_in], columns='cycle',
                         values=ref_sample['cycle'] + np.arange(look_ahead)/trmaxrul)
util.plot_series(hazard, figsize=figsize, title=f'Hazard function at cycle {ref_sample["cycle"]*trmaxrul:.0f} (look ahead: {look_ahead})')
```
![[hazardchangingt.png]]
The model has learned that time has an effect on 𝜆
In hazard ho gli estimated HAZARDS computati con la stessa $x_{k, t}$ come input e variando soltanto il _t_.
Calcolo poi l'ESTIMATION per ogni punto nel futuro secondo la formula qui sopra $S(t+n)/S(t)$ per ogni n in 0..300:
```python
survival = pd.Series(data=np.cumprod(1-hazard))
util.plot_series(survival, figsize=figsize, ylim=(0,1),
                 title=f'Conditional survival function at cycle {ref_sample["cycle"]*trmaxrul:.0f} (look ahead: {look_ahead})')
```
e ho come risultato:
![[conditionalsurvivaltime.png]]
Dopo 10 steps ho LA PROBA A 0 praticamente.

POSSO ANCHE CALCOLARE cosa succedera' tra 30 steps IN REAL TIME. quindi praticamente per ogni punto posso fare FORECAST NEL FUTURO di 30 steps. Appena noto che la probabilita' si sta abbassando ALLORA posso prevenire la cosa (visto che sto facendo forecast, posso PREVENIRE la cosa).
C'e' comunque da prendere queste probabilita' nel FUTURO come OTTIMISTE (perche' si usa x_k,t sempre) e c'e' da tenere in considerazione che sono motlo NOISY proprio per questo motivo e anche perche' sto guardando molto nel futuro.