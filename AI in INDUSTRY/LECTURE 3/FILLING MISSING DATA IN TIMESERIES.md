## From sparse to dense indexes

```python
import numpy as np
delta = data.index[1:] - data.index[:-1]
pd.Series(delta).value_counts()
```
In this way I see the counts of the different timegaps.
*timestamp
0 days 00:05:00    1754
0 days 00:10:00     340
0 days 00:15:00     106
0 days 00:20:00      37
0 days 00:04:00      26
0 days 00:25:00      22
0 days 00:06:00      18
0 days 00:30:00       9
0 days 00:35:00       8
0 days 00:11:00       7*

**By far the most common value is 5 minutes**

* Some values are not multiples of 5 minutes (e.g. 4, 6, 11 minutes)
* I.e. they are _out of alignment_
* 
**Therefore, first we need to _realign_ the original index**

This is also called _resampling_ (or _binning_), and can be done in pandas with:

```python
DatetimeIndex.resample(rule=None, ...)
```

* `rule` specifies the length of each individual interval (or "bin")

**Resample is an iterator: we need to choose what to do with each bin**

E.g. compute the mean, stdev, take the first value. In our case we take the mean of the values that fall in that bin (if one bin is empty then the value is NaN).

```python
ddata = data.resample('5min').mean()
ddata.head()
```

## How to measure the efficacy of our FILLING methods
**We will now discuss a few simple approaches to deal with missing values**

We will use _partially synthetic data_

* We will focus on specific (and mosly intact) sections of our series
* The we will remove values artificially
* ...And measure the accuracy of our filling approaches via the Root MSE

$$
\mathit{RMSE} = \sqrt{\frac{1}{n}\sum_{i = 0}^n (x_i - \hat{x}_i)^2}
$$

Where $x_i$ is a value from the filled series and $\hat{x}_i$ the ground truth

* $x_i = \hat{x}_i$ if no value is missing
* Hence, any MSE difference is entirely due to missing values

## Forward/Backward Filling
**The easiest approach for missing values consists in _replicating nearby observations_**
* _Forward filling:_ propagate forward the last valid observation
* _Backward filling:_ propagate backward the next valid observation

**An important observation:**
* When filling missing values, _we have access to the whole series_
* ...So we can reason _both forward and backward_

**Forward/backward filling are simple methods, but they can work well**
* Rationale: most time series have a certain "inertia"
* ...I.e.: a strong level of local correlation
* For this reason (e.g.) _the last observation is often a good predictor_ for the next one

**Forward and backward filling are pre-implemented in pandas**
They are available through the `fillna` method:
```python
DataFrame.fillna(..., method=None, ...)
```
* `fillna` replaces `NaN` values in a `DataFrame` or `Series`
* The `method` parameter can take the values:
  * "pad" or "ffill": these correspond to forward filling
  * "backfill" or "bfill": these correspond to backward filling
They are generally applied to datasets with a dense index
* Remember that our benchmark dataset already has a dense index

We apply the evaluation of how much the filling method is good with forward/backward filling:
```python
nan_mask = ddata['value'].isnull()
#ddata_mv has 30 random points removed
# forward filling
ffseries = ddata_mv.fillna(method='ffill')
ffseries[nan_mask] = np.NaN # We empty the values that were originally empty
# backward filling
bfseries = ddata_mv.fillna(method='bfill')
bfseries[nan_mask] = np.NaN # We empty the values that were originally empty

# we compute the RMSE between ddata and ffseries, without considering the points that were ALREADY missing (the ones for which the ground truth is NaN).
rmse_ff = np.sqrt(mean_squared_error(ddata[~nan_mask], ffseries[~nan_mask]))
rmse_bf = np.sqrt(mean_squared_error(ddata[~nan_mask], bfseries[~nan_mask]))
print(f'RMSE for forwad filling: {rmse_ff:.2f}, for backward filling {rmse_bf:.2f}')
```
*RMSE for forwad filling: 1.33, for backward filling 0.87*
* In this case backward filling seems to work better
* The results are of course application-dependent

* Forward/backward filling tend to work well for _low variance_ sections
* ...And conversely work worse for _high variance_ sections

## Geometric interpolation filling
**A few more options are available via [the `interpolate` method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html)**
```python
DataFrame/Series.interpolate(method='linear', ...)
```

The `method` parameter determines how NaNs are filled:

* "linear" uses a linear interpolation, assuming uniformly spaced samples. Si fa una linea tra l'observation valida a sx e quella valida a dx.
* "time" uses a linear interpolation, but supports non-uniformly spaced samples
* "nearest" uses the closest value. Usa FORWARD filling se il punto piu vicino e' a sx, BACKWARD filling se quello piu vicino e' a dx
* "polynomial" uses a polynomial interpolation
* Even "ffill" and "bfill" are available
NOTE: all the above methods take into account the past and the future. In FORWARD/BACKWARD filling only one of them.
Both "polynomial" and "spline" require to specify the additional parameter `order`

* E.g. `df.interpolate(method='polynomial', order='3')`

```python
args = [{'method':'linear'}, {'method':'time'}, {'method':'nearest'},
        {'method':'polynomial', 'order': 2}, {'method':'spline', 'order': 4}]
filling_res = {}
for a in args:
    filling_res[a['method']] = ddata_mv.interpolate(**a)
    rmse = np.sqrt(mean_squared_error(ddata[~nan_mask], filling_res[a['method']][~nan_mask]))
    print(f'RMSE for {a["method"]}: {rmse:.2f}')
```
*RMSE for linear: 0.98
RMSE for time: 0.98
RMSE for nearest: 1.35
RMSE for polynomial: 1.10
RMSE for spline: 1.03*
* "linear" and "time" are equivalent (we have uniformly-spaced samples)
* "polynomial" is the most complex, and in this case also the worst

All perform _worse_ than backward filling (at least in this case)!

* Linear filling works well for series with _slower dynamics_ 
* ...But does not work well for series that _faster dynamics_

* Nearest filling is a _compromise between forward and backward filling_
* ...And in our case its performance is the average of the two

* Polynomial interpolation relies on nearby values to fit a polynomial
* _High-order polinomial often vary too much_ and work less well

* Spline interpolation relies on [piecewise polynomial curves](https://en.wikipedia.org/wiki/Spline_(mathematics))
* ...And it's often _more robust than polynomial interpolation_

# A better approach: GAUSSIAN PROCESSES
ASSUMPTIONS OF THE MODEL
1. **Assumption 1 (intuitively)**
* _For every value_ of the index variable the distribution is _Gaussian_
* Therefore it can be described by a _mean_ and a _standard deviation_
2. **Assumption 2 (intuitively)**
* The stdev depends on the distance between a point and the observations
* So it will be _low_ when we are _close_ to the observations, _high_ when we are _far_ 

GP e' una collezione di random variables. A ogni index $\bar{x}$ infatti corrisponde una random variable $\bar{y}_\bar{x}$. Quindi ogni  $\bar{y}_\bar{x}$ rappresenta una probabilita' di distribuzione. Quindi alla fine ho che GP e' una **MULTIVARIATE DISTRBUTION** in cui ogni dimensione di questa multivariate distribution e' una univariate distribution rappresentata dalla random variable $\bar{y}_\bar{x}$ per ogni index input  $\bar{x}$. Note that the random variables $\bar{y}_\bar{x}$ are correlated to each others.

We already know that the PDF for a Multivariate Normal Distribution is defined via:
* A (vector) mean $\mu$
* A covariance matrix $\Sigma$
### GP at training and inference time
By recentering we can assume $\mu = 0$ for each random variable  $\bar{y}_\bar{x}$ (we can standardize each of them), meaning that _knowing $\Sigma$ is enough_.
**Therefore, if we know $\Sigma$ we can easily compute**
* The joint density $f(\bar{y}_{\bar{x}})$ for a set of observations. It is computed by doing: 
$$ f(\bar{y}_{\bar{x}} \mid \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k \lvert \Sigma \rvert}} \exp\left(-\frac{1}{2} (\bar{y}_{\bar{x}} - \mu)^T \Sigma^{-1} (\bar{y}_{\bar{x}} - \mu)\right) $$
con $\bar{y}_{\bar{x}}$ che contiene pero' stavolta i PUNTI CONOSCIUTI (quelli che insomma HO a mia disposizione).
* The conditional density $f(y_x \mid \bar{y}_{\bar{x}})$ of an observation $y_x$, given $\bar{y}_{\bar{x}}$. Questa infatti non e' nient'altro che $$
f(y_x \mid \bar{y}_{\bar{x}}) = \frac{f(y_x, \bar{y}_{\bar{x}})}{f(\bar{y}_{\bar{x}})}
$$
(NOTA PERO': come faccio a calcolare $f(y_x, \bar{y}_{\bar{x}})$ con una covariance matrix statica? Che quindi non e' estendibile in grandezza? Ci torno qua sotto...)
**We need the joint density to perform training!**
...Because in practice we _don't know $\Sigma$_
* Therefore we'll assume that $\Sigma$ is a _parameterized function $\Sigma(\theta)$_
* ...And we can optimize the parameters $\theta$ for maximum likelihood
**Formally, given a set of training observations $\bar{y}_{\bar{x}}$**
...We can calibrate the parameters by solving a problem in the form:
$$
\mathop{\arg\max}_{\theta} f(\bar{y}_{\bar{x}}, \theta)
$$
* Here we are not using a **product of probabilities** over the training set
* ...Because the $y_x$ variables are correlated
* No worries: since we have the joint PDF, we use it directly
POSSO QUINDI FARE MLE praticamente, per rendere l'estimator _f_ sempre piu' buono sui punti in input.
**PERO' c'e' un problema (questa cosa non puo' funzionare):**
Say that we a covariance matrix $\Sigma$ for a set of observations $\bar{y}_{\bar{x}}$

* Now we want to perform inference for an input value $x$
* Formally: we want to compute $f(y_x \mid \bar{y}_{\bar{x}})$
**In principle, we can use the formula:**

$$
f(y_x \mid \bar{y}_{\bar{x}}) = \frac{f(y_x, \bar{y}_{\bar{x}})}{f(\bar{y}_{\bar{x}})}
$$

* By using our $\Sigma$, we can easily compute $f(\bar{y}_{\bar{x}})$
* ...But what about $f(y_x, \bar{y}_{\bar{x}})$?

**Our $\Sigma$ refers to our set of observed variables $\bar{y}_{\bar{x}}$**

Therefore, if we have $n$ variables, our matrix will be $n \times n$

$$\Sigma_{\bar{x}} = \left(\begin{array}{cccc}
\sigma_{\bar{x}_1,\bar{x}_1} & \sigma_{\bar{x}_1,\bar{x}_2} & \cdots & \sigma_{\bar{x}_1,\bar{x}_n} \\
\sigma_{\bar{x}_2,\bar{x}_1} & \sigma_{\bar{x}_2,\bar{x}_2} & \cdots & \sigma_{\bar{x}_2,\bar{x}_n} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{\bar{x}_n,\bar{x}_1} & \sigma_{\bar{x}_n,\bar{x}_2} & \cdots & \sigma_{\bar{x}_n,\bar{x}_2}
\end{array}\right)$$

* In every cell we have the covariance for variables $\bar{y}_{\bar{x}_i}$ and $\bar{y}_{\bar{x}_j}$
* With this matrix, we can compute $f(\bar{y}_{\bar{x}})$
**However, $f(y_x, \bar{y}_{\bar{x}})$ refers to _one more variable_**

Meaning that it will be specified via an $(n+1) \times (n+1)$ matrix!

$$\Sigma_{x,\bar{x}} = \left(\begin{array}{ccccc}
\color{red}{\sigma_{x,x}} & \color{red}{\sigma_{x,\bar{x}_1}} & \color{red}{\sigma_{x,\bar{x}_2}} & \color{red}{\cdots} & \color{red}{\sigma_{x,\hat{x}_2}} \\
\color{red}{\sigma_{\bar{x}_1,x}} & \sigma_{\bar{x}_1,\bar{x}_1} & \sigma_{\bar{x}_1,\bar{x}_2} & \cdots & \sigma_{\hat{x}_1,\hat{x}_n} \\
\color{red}{\sigma_{\bar{x}_2,x}} & \sigma_{\bar{x}_2,\bar{x}_1} & \sigma_{\bar{x}_2,\bar{x}_2} & \cdots & \sigma_{\bar{x}_2,\bar{x}_n} \\
\color{red}{\vdots} & \vdots & \ddots & \vdots \\
\color{red}{\sigma_{\bar{x}_n,x}} & \sigma_{\bar{x}_n,\bar{x}_1} & \sigma_{\bar{x}_n,\bar{x}_2} & \cdots & \sigma_{\bar{x}_n,\bar{x}_2}
\end{array}\right)$$

* Assuming that $\bar{y}_{\bar{x}}$ are the training observations
* ...We could define $\sigma_{\bar{x}_1, \bar{x}_2}$ at training time

**But how do we define _the new covariances_, i.e. those related to $y_x$?**
Come faccio praticamente a estendere la covariance matrix a inference time?
**We assume that covariance can be _built from a set of inputs_**
Let $\bar{x}$ refer now to a vector of values of our input index variables.
* Given two random variables $\bar{y}_\bar{x_i}$ and $\bar{y}_\bar{x_j}$
* We specify their covariance via parameterized _kernel function_ $K_\theta(\bar{x}_i, \bar{x}_j)$
* $K$ typically depends on the distance between input values
* Nota che la kernel function e' sui VALORI degli indici (quindi nel nostro caso prendera' come inputs timestep values).
Quindi ho che:
**Given any finite set of variables $\{y_{x_1}, \ldots y_{x_n}\}$, the covariance matrix is:**

$$\Sigma = \left(\begin{array}{cccc}
K_\theta(x_1, x_1) & K_\theta(x_1, x_2) & \cdots & K_\theta(x_1, x_n) \\
K_\theta(x_2, x_1) & K_\theta(x_2, x_2) & \cdots & K_\theta(x_2, x_n) \\
\vdots & \vdots & \vdots & \vdots \\
K_\theta(x_n, x_1) & K_\theta(x_n, x_2) & \cdots & K_\theta(x_n, x_n) \\
\end{array}\right)$$

...Which we can computed based on the input (and the parameters) alone!
IN STO MODO POSSO ESTENDERE $\Sigma$ a un nuovo $x$ a inference time. Infatti 
at _training time_** what happens is that
* Pick a _parameterized_ kernel function $K_{\theta}(x_i, x_j)$
* Collect training observations $\bar{y}_{\bar{x}}$
* Optimize the kernel for maximum likelihood (e.g. via **gradient descent**)

Both the parameters $\theta$ _and the observations $\bar{y}_\bar{x}$_ are stored in the model
(solo queste due, perche' OGNI volta, anche durante il training e poi a inference time, la covariance matrix viene RICOSTRUITA a partire da $\theta$ e $\bar{y}_\bar{x}$)

* This is similar to what we have in Kernel Density Estimation
**At inference time invece:**
* Given a new input (i.e. index) value $x$
* We obtain the covariance matrix $\Sigma_{\bar{x}}$
* We obtain the covariance matrix $\Sigma_{x, \bar{x}}$
Con ste due covariance matrices possiamo calcolare la PDF (guarda come si calcola la PDF di una multivariate gaussian dist qua sopra) PRIMA con sia x che $\bar{x}$ usando la  $\Sigma_{x, \bar{x}}$, e poi solo con $\bar{x}$ usando la  $\Sigma_{\bar{x}}$. 
...And with this we can completely characterize $f(y_x, \mid \bar{y}_\bar{x})$ 

### GP in SCIKIT-LEARN
**We will start with a simple _Radial Basis Function_ (i.e. Gaussian) kernel**

$$
K(x_i, x_j) = e^{-\frac{d(x_i, x_j)^2}{2l}}
$$

The covariance _decreases with the (Euclidean) distance_ $d(x_i, x_j)$:

* Intuitively, _the closer the points, the higher the correlation_
* The $l$ parameter (_scale_) control the rate of the reduction (se e' grande allora il decay della correlation sara' piu' lento, piu' piccolo allora il decay sara' piu' veloce)
**Here's how to use an RBF kernel in scikit-learn**
```
kernel = RBF(1, (1e-2, 1e2))
```
**The RBF kernel has a single parameter, representing its _scale_**
The extra (tuple) parameter represents a pair of _bounds_
* During training, only values within the boundaries will be considered (serve per boundare il search space)

```python
from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(y_tr.index.values.reshape(-1,1), y_tr.values) # needs 2D input
```
La target function e' in sto caso
```python
f = lambda x: x * np.sin(2*np.pi*x) + x # target function
```
e y_tr contiene 15 punti calcolati usando sta target function. Al GP vengono dati come input i 15 timesteps e i 15 valori della funzione corrispondenti.

* Training uses Gradient Descent to maximize the likelihood of the training data
* _Restarts_ are needed to mitigate issues due to local optima

La prediction per ogni punto dell'INTERO dataset x e' calcolata cosi:
```python
xp, std = gp.predict(x.reshape(-1,1), return_std=True)
xp = pd.Series(index=y.index, data=xp)
std = pd.Series(index=y.index, data=std)
```
Per ogni punto e' ritornata una media e una standard deviation (non ho point estimates come predictions, MOLTO PIU INFORMATIVO).

xp, std rappresentano una **fully characterized conditional distribution**.
The distribution is conditional on:
* The input value $x$ (passed at inference time)
* All the training observations (stored in the model)
* Quindi e' PDF di (x| training obs) come avevamo gia detto sopra.
![[reconstruction.png]]
Questo e' il risultato con 15 punti. NON male. NOTA come GP negli EDGES performi un po' di merda, perche' non ha molti punti negli edge, quindi e' piu' incerto (puo' solo usare i punti a DX nel caso dell'edge sinistro o a SX nel caso di edge destro).
DI BASE HO QUINDI UN CONFIDENCE INTERVAL TROPPO OTTIMISTICO SUGLI EDGES IN PARTICOLARE.
come risolvo?

SE GUARDO ALLA TRUE FUNCTION, POSSO CHEATTARE UN PO' E CAPIRE DI CHE TIPO DI KERNEL HO BISOGNO PER FARE UNA PREDIZIONE DEI MISSING VALUES MIGLIORE:
Abbiamo di sicuro
**some _noise_ (non ho ben capito dove veda il noise nella true function pero' va bene), a _period (dovuto al fatto che comunque ci sono ste curve che si ripetono. Posso usare autocorrelation plots, invece di cheattare (in una IRL application non posso vedere la true function) per dare una buona stima inziale della lunghezza del period. Poi sara' il GP a ottimizzare con GD)_, and a _trend_(SAREBBE la tendenza dei valori a CRESCERE over time, infatti ho onde che diventano SEMPRE PIU' ALTE, quindi y values che aumentano)**
Quindi faccio le seguenti modifiche:
```
from sklearn.gaussian_process.kernels import DotProduct
kernel = WhiteKernel(0.1, (1e-2, 1e2))
kernel += ConstantKernel(1, (1e-2, 1e2)) * RBF(1, (1e-2, 1e2))
kernel += ExpSineSquared(1, 1, (1e-1, 1e1), (1e-1, 1e1))
kernel += DotProduct(1, (1e-2, 1e2))
```
WhiteKernel fa 
$$
K(x_i, x_j) = \sigma^2 \text{ iff } x_i = x_j, 0 \text{ otherwise}
$$
* The only parameter of `WhiteKernel` represents the noise level $\sigma^2$
* A small noise level prevents overfitting
* ...But too much noise leads to useless predictions!

**`ExpSineSquared` captures the period:**

$$
K(x_i, x_j) = e^{-2 \frac{\sin^2 \left(\pi \frac{d(x_i,x_j)}{p}\right)}{l^2}}
$$
* The correlation grows if the distance is close to a multiple of the period $p$
* The scale parameter $l$ controls the rate of decrease/increase
* In the implementation, the first parameter is $l$ and the second $p$

**`DotProduct` (somewhat) captures the trend:**

$$
K(x_i, x_j) = \sigma^2 + x_i x_j
$$

* The larger the $x$ values, the larger the correlation (se ci pensi se entrambi sono grandi allora avro' una correlation alta che era proprio quello che volevo ottenere, se entrambi bassi una correlation bassa pero', il che e' un po' SUSSY no? Ricordo che non era il massimo sto dotproduct but still...)
* This allows the distance from the mean (which is zero) to grow
* The $\sigma$ parameter controls the base level of correlation
* Unlike all kernels so far `DotProduct` is _not translation-invariant_
![[reconstructionFinalResult.png]]
This is the result.
Ad ogni modo GP non e' bravissimo a catturare i trends. Ci riescono Linear Regressor, NN, decision trees addirittura ma non GP. Volendo avrei potuto (sarebbe stato meglio) predictare con un linear regressor il trend e sommare il risultato al kernel costruito.

### GP IN THE USE CASE (filling missing values in TRAFFIC DATA)
Se guardo il data noto che 
![[trafficData.png]]
1. Ho *local correlation*, ma questa non e' cosi' grande perche' ho MOLTO noise. 
2. C'e' la presenza di un periodo che pero' NON e' molto regolare.
3. NON c'e' nessun *trend*, quindi NON c'e' nessuna tendenza a crescere o a diminuire over time.
**IMPORTANTE** vedere i dati mi aiuta a capire quale kernel USARE.

**SHOULD I USE ALL THE AVAILABLE DATA FOR TRAINING?**
NO, perche' cosi facendo
1. non avrei un validation set
2. ho un grande gap nel centro quindi non ha molto senso
3. il training time diventa altissimo
Ha molto piu' senso concentrarsi su una parte del dataset in cui ci sono magari pochi missing values 
Se pero' uso solo una frazione del dataset ho che:
1. rischio di non catchare informazioni che avrei catchato se avessi avuto una frazione piu' grande (tipo la presenza del periodo). Voglio che questo periodo SIA PRESENTE nel training set in modo tale che il GP possa impararlo. 
Alla fine decido di prendere una sezione del dataset di piu' di UNA SETTIMANA.

WAIT
**We have no ground truth: how are we going to evaluate the kernels?**
We will use the same trick we used before:
* We will _focus on a portion_ of our sequence
    * ...One with relatively few missing values
* Then we will _artificially remove_ part of the data points
    * This will form the ground truth for our evaluation
Main idea: use _part of our data as a validation set_
**Which quality metric?**
* Thanks to the availability of confidence intervals...
* ...We can compute the _likelihood_ of our validation set!
* Using the MSE would do the same, only with more assumptions
In verita' il prof. consiglia di utilizzare una misura chiamata z-score che calcola a quante **standard deviations** il punto REALE sta dalla mean del punto predictato dal GP (e' un tipo di distanza che tiene in considerazione l'STD).
Metto un noise kernel e un period kernel cosi
```
kernel = WhiteKernel(1e-3, (1e-4, 1e-1))
kernel += ConstantKernel(1, (1e-2, 1e2)) * RBF(1, (1e-1, 1e1))
kernel += ConstantKernel(1, (1e-2, 1e2)) * ExpSineSquared(1, 2000, (1e-1, 1e1), (1900, 2100))
```

Ho che il valore della likelihood sul validation set (che sembra essere un z score tra l'altro dal notebook da quello che sembra) diminuisce se utilizzo un noise + RBF + period kernel, invece di un solo noise + RBF kernel, ma I confidence intervals sono TROPPO GRANDI.

Voglio ora pero' far le prediction SU TUTTO IL DATASET. come faccio? Non posso usare SOLO i punti del training set, sono TROPPO LONTANI a altri punti del dataset che devo predictare (la prediction sara' terribile no? Farei extrapolation MOOOLLTO nel passato).
**We now need to obtain predictions _for the whole series_**

We would prefer to **_avoid training again the kernel_ parameters**
* The large number of missing value may be problematic
* ...And the training time would be very large

...But we _also_ really wish to use _all available observations_
* ...Not just those considered when training the kernel

**With Gaussian Processes, we can do both**
There is _no need to train again the kernel_ every time new observations arrive
* We can build a new $\Sigma$ matrix using _the new observations_ and _the old kernel_ (GENIALE, solo il GP te puo' far fare sta cosa)
```python
gp2 = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None)
```
per trasferire.
Passing `optimizer=None` will _disable optimization_ at training time

**So that calling `fit` will just take into account _a new set of observations_** (non c'e' NESSUN TRAINING QUANDO CHIAMO FIT adesso).
![[fillmissingvalues1.png]]
2 PROBLEMS:
1. The std values are WAY TO NEGATIVE (as they are way too positive) and this doesn't make a lot of sense.
BUT THIS IS THE NATURE OF GP WE CANNOT DO ANYTHING ABOUT IT.
2.  **The confidence intervals are large _even for the NIGHT HOURS (non ha senso)_!** We can instead work on this 

There are two reasons for the last point:
* There are _fewer samples_ at nighttime
    * As we get far from the samples the confidence drops (quickly, in our case)
* No traditional GP kernel can represent _input-dependent variance_
    * All kernels are about covariance, not variance
    * The lone exception is the `WhiteKernel`, which is not input dependent
    * Cioe' vorrei un kernel che prende in considerazione il timestep in cui sto: quindi SE il timestep rappresenta un'ora notturna allora VOGLIO MENO VARIANZA (ma non esiste sto kernel).
**Can we deal with this issues?**
Posso 
1. O implementare un CUSTOM KERNEL
2. O usare un ENSEMBLE MODEL (combino piu' modelli insieme banalmente)
HO diversi tipi di ensembe models:
1. VOTING: come in random forests (in cui il risultato e' quello piu' votato dai modelli)
2. ADDING: come in caso di ada boosting (in cui il risultato e' dato dalla sum dei risultati outputtati dai modelli e in cui il risultato di un modello viene usato come input del modello successivo).
3. SELECTING: come quello usato con KDE nello scorso CAPITOLO (in cui il risultato e' dato DA UN SOLO MODELL SCELTO a SECONDA del timestep in cui sono)
Ne introduco uno del SECONDO TIPO: ADDING ENSEMBLE
Di sicuro quello che introduco non e' del tipo 1 perche' dovrei trovare diversi estimators che ritornano un **confidence interval** e praticamente esiste solo GP quindi non e' possibile.
Il tipo 3 potrebbe funzionare ma il mio dataframe ha troppi pochi punti (trainerei diversi modelli su pochissimi punti)
Ne introduco uno del SECONDO TIPO: ADDING ENSEMBLE
Uso un secondo modello che pensera' a gestire l'**input dependent variance**.

**Come si trainano due modelli in un ensemble di questo tipo?**
Mettiamo che questi due modelli sono di questo tipo: $g(x, \theta)$ e $h(x, \gamma)$
Posso o far coevolvere $g(x, \theta)$ e $h(x, \gamma)$ insieme o trainare prima $g(x, \theta)$ e poi $h(x, \gamma)$. Scelgo il secondo approccio.
Nota che io voglio che $g(x, \theta)$ e $h(x, \gamma)$ mi approssimino insieme il target $y$, quindi punto a qualcosa del tipo:
$g(x, \theta)+ h(x, \gamma) = y$
Quando traino $g(x, \theta)$, lo traino per predictare $y$, come se $h(x, \gamma)$ non ci sia.
Una volta finito il training di $g(x, \theta)$ e ho trovato $\theta ^ *$  ho che 
$h(x, \gamma) = y - g(x, \theta ^ *)$ che sarebbe il **residual**. traino h quindi per stimare il residual. Posso seguire questo approccio solo se g e h sono modelli RADICALMENTE diversi.
Nella sezione di prima potevo infatti usare g linear regressor e h GP. Dopo aver fittato il linear regressor, avrei potuto sottrarre la prediction del linear regressor g a y e trainare il GP sul residual (nel residual rimangono SOLO le caratteristiche non catturate dal Linear Regressor e ORA catturabili dal GP).

NEL MIO CASO DI ORA, ho che sommare NON funzionerebbe, perche' $$
\mathit{Var}(x + \alpha) = \mathit{Var}(x)
$$ mentre moltiplicare si', perche' 
$$
\mathit{Var}(\alpha x) = \alpha^2 \mathit{Var}(x)
$$
QUINDI **our model will become _the product of two models_**

Formally, we will have:

$$ 
g(x, \lambda) f(x, \theta)
$$

* $f$, with parameters $\theta$ will be a Gaussian Process
* $g$, with parameters $\lambda$ will be our _variance model_ (or _standard deviation model_)

**On the training set, we wish to have:**

$$
g(x_i, \lambda) f(x_i, \theta) \simeq y_i \quad\Rightarrow\quad f(x_i, \theta) \simeq \frac{y_i}{g(x_i, \lambda)}
$$

* The Gaussian Process will need to learn a series with a variance _altered by $g$_
* The variance of each point $y_i$ will be divided by $g(x_i, \lambda)^2$
NEL NOSTRO CASO, non c'e' bisogno di attuare prima un training su g e poi uno su f col residual ma invece possiamo direttamente andare a fare il training su f col residual, perche' scegliamo g come un MAPPING STATICO legato all'ora del giorno.

**We now need to choose our variance model $g$**

* Since we have discrete time and a natural period (a week)
* ...We could use a simple map (time of the week $\rightarrow$ standard deviation)

**Let's add a "hour of the week" information to our data:**

The chosen time unit is actually irrelevant

Computo una lookup table in cui si prendono in considerazione gaps di 1 ora e si calcola la standard deviation per quei gaps. Questo pero' porta a un mapping molto spigoloso e poco smooth (ho il cambiamento del valore della std SOLO dopo un'ora).
Quindi il Prof. applica un UPSAMPLING (non so in che senso), che porta a molti valori NaN, e poi fa linear interpolation. Dopo di che applica uno smoothing filter per rendere il tutto meno spigoloso.

Una volta che ho sta STD MAPPATA per ogni punto del dataset faccio y / std  e utilizzo UNA PARTE DI questo residual (che e' quella zona corrispondente a UNA SETTIMANA) per trainare il GP.
Estendo il GP a TUTTI I PUNTI (senza riottimizzare niente come detto sopra) e faccio predicitons. MOLTIPLICO poi le predictions (che consistono in mean e std vectors) per L'STD MAPPATA in modo da ottenere i veri risultati. 
Li plotto e il risultato e' : 
![[resultsensembel.png]]

ORA, PER FILLARE I VALORI POSSO:
1. o usare max (0, mean ) 
2. o usare max(0, SAMPLE given the mean and the std) 
per ogni punto che era nullo nel dataset iniziale, e di cui ora ho a disposizione mean e std predictati.
Questo e' il risultato del secondo approccio: 
![[sampleresult.png]]
NOT FUCKING BAD!