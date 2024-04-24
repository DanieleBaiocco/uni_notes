
**HPC is (somewhat) distinct from cloud computing**

* Cloud computing is mostly about running (and scaling) _services_
* ...HPC is all about _performance_

Typical applications: simulation, massive data analysis, training large ML models

**HPC systems follow a _batch computation_ paradigm**

* Users send _jobs_ to the systems (i.e. configuration for running a program)
* Jobs end in one of several _queues_
* A _job scheduler_ draws from the queue
* ...And dispatches jobs to computational _nodes_ for execution
### How to display multivariate series?
1. **Approach #1**: showing _individual columns_
```python
tmp = pd.Series(index=hpc['timestamp'], data=hpc[inputs[0]].values)
util.plot_series(tmp, figsize=figsize)
```
 ![[individualcolumn2.png]]
 **NOTE** THERE ARE MISSING TIMESTEPS.
 1. **Approach #2**: obtaining _statistics_
```python
 hpc[inputs].describe()
```

3. **Approach #3**: standardize, then use a _heatmap_
```python
hpcsv = hpc.copy()
hpcsv[inputs] = (hpcsv[inputs] - hpcsv[inputs].mean()) / hpcsv[inputs].std()
util.plot_dataframe(hpcsv[inputs], figsize=figsize)
```
Standardization is done because heatmap works with values between 0 and 1. If the value is close to 1 the color is warmer, if it is closer to 0 it is cloder.![[heatmap.png]]
In the x axis there are the timesteps, in the y axis the features. The color represents how much the value of that point is high.
From the heatmap I note that there are clear and neat **SEPARATION LINES**. This is due to the fact that the jobs **DON'T COMPETE** in the usage of the CPU of the HPC. I have only **ONE JOB AT A TIME**. The separation lines are lines in which one JOB stops and another starts. 

###  Anomalies
**There are three possible configurations of the frequency governor:**

* Mode 0 or "normal": frequency proportional to the workload
* Mode 1 or "power saving": frequency always at the minimum value
* Mode 2 or "performance": frequency always at the maximum value

**On this dataset, this information is known**

...And it will serve as our ground truth

* We will focus on discriminating normal from non-normal behavior
* I.e. we will treat both "power saving" and "performance" cases as **anomalous**

**Detecting them will be _challenging_**

* Since the signals vary so much when the running job changes
The anomalies are not present in the first part of the dataset, as we can see from
![[anomalies.png]]

## KDE APPROACH
First, we _standardize_ the data again, based on _training information alone_, because we don't want to accidentally exploit test set information
* The training set separator was chosen so as not to include anomalies
Then we separate in train, validation and test data.
We do GridSearchCV on the train data, with MLE on the folds, in order to get the best bandwidth value.
We then score the samples with the best bandwidth and the result is these signals: _**
![[signals2.png]]
**There is _a good match_ with the anomalies, but also some _spurious peaks
* This is mostly due to the large variations due to job changes

**We then need to define the threshold, but for that we need a cost model**

Our main goal is to detect anomalies, not anticipating them

* Misconfigurations in HPC are usually not critical 
* ...And cause little issue, unless they stay unchecked for very long

**We will use a simple _cost model_:**
* $c_{\mathit{alarm}}$ for false positive (erroneous detections)
* $c_{\mathit{missed}}$ for false negatives (undetected anomalies)
* Detections are fine as long as they are within $\mathit{tolerance}$ units from the anomaly

We then find the best threshold, as we did in the previous Lecture. The only difference is that this time we have a different cost function and that we perform this optimization **ONLY** on the validation set (without merging train+validation set)
```python
th_range = np.linspace(1e4, 2e5, 200)
th_kde, val_cost_kde = util.opt_threshold(signal_kde[tr_end:val_end],
                                        valdata['anomaly'],
                                        th_range, cmodel)
print(f'Best threshold: {th_kde:.3f}')
tr_cost_kde = cmodel.cost(signal_kde[:tr_end], hpcs['anomaly'][:tr_end], th_kde)
print(f'Cost on the training set: {tr_cost_kde}')
print(f'Cost on the validation set: {val_cost_kde}')
ts_cost_kde = cmodel.cost(signal_kde[val_end:], hpcs['anomaly'][val_end:], th_kde)
print(f'Cost on the test set: {ts_cost_kde}')
```
*Best threshold: 148442.211
Cost on the training set: 0
Cost on the validation set: 263
Cost on the test set: 265*

### Limits of KDE
**KDE-based approaches work well, but have some _issues_**
First, KDE itself runs into trouble with _high-dimensional data_:
* With a larger dimensionality, _prediction times_ grows...
* ...And **_more data_ is needed to obtain reliable results**
* This is basically the curse of dimensionality: with more dimensions, I have zones in which I have no points from the dataset, therefore it generalizes poorly.
Second, KDE has trouble with _large training sets_
* The more the samples in the training set
* ...**The more the terms to be summed to obtain a density**
Third, KDE gives you _nothing more_ than an anomaly signal
* Determining the cause of the anomaly is up to a domain expert
* This is ok in low-dimensional spaces, but _harder on high-dimensional ones_
KDE in this case has HIGH VARIANCE (the output changes drastically with few changes in the input data) and LOW BIAS (the model cannot do anything if It goes in a region with few samples).

## Gaussian Mixture Models APPROACH
**We'll start by focusing on the scalability issues**
We have established that KDE has trouble with:
* Large dimensional datasets
* Large number of training examples

**KDE makes no attempt to "compress" the information from the training data:**
* The size of a KDE models grows directly with the training set size
It's time to introduce a new density estimation technique

**In particular, we'll now switch to using _Gaussian Mixture Models (GMMs)_**
A GMM describes a distribution via a _weighted sum of Gaussian components_
* The model size depends on the dimensionality and on \#components
* The \#components can be chosen, to control the bias/variance trade-off
**Formally, we assume data is generated by the following probabilistic model**
$$
X_Z
$$
* $Z$ e $X_k$ are both random variables
* $Z$ represents the index of the component that generates the sample
* $X_k$ follows a multivariate Gaussian distribution
In other words, a GMM is **_a selection-based ensemble_**, as It was our multiple-KDE estimator.

**The PDF of a GMM is given by:**
$$
g(x, \mu, \Sigma, \tau) = \sum_{k = 1}^{n} \tau_k f(x, \mu_k, \Sigma_k)
$$
* $f$ is the PDF of a multivariante Normal distribution
* $\mu_k$ is the (vector) mean and $\Sigma_k$ the covariance matrix for the $k$-th component
* $\tau_k$ corresponds to $P(Z = k)$
So we basically specify a point x for which we would like to understand the PDF. And it is basically a weighted average of the PDFs of the multivariate Normal distributions (that are the k **components**).

In our case with 2 components we have this result, that is generated at random (not from the data, just for understanding Gaussian Mixtures):
![[gaussianmixture.png]]
This has 
```
tau: [0.67771161 0.32228839]
mu [[0.65362832 0.09209144]  [0.32779662 0.97758091]]
sigma [array([[0.12356578, 0.144611  ],        [0.144611  , 0.27972214]]), array([[0.16040371, 0.07719013],        [0.07719013, 0.08245494]])]
```
* Our example has two components, each with its own mean and covariance
* One component is slightly less prevalent than the other
### Sampling From Gaussian Mixture
**When we want to _sample_ from a GMM**
* First we need to sample the $Z$ variable
* Then we sample from the corresponding multivariate distribution
```python
train_x, train_z = gt.sample(1000, seed=42)
test_x, test_z = gt.sample(1000, seed=42)
```
train_z contains an array of 1000 values, in which for each value there is the index of the multivariate distribution (the component) from which that value was sampled.
We **ASSUME** basically that the data was sampled from these multivariate gaussian distributions, that have a tau probability associated. 
We need a way to find the params that best describe the data in terms of this Gaussian Mixture model.

### Training a GMM
**We can train a GMM to _approximate other distributions_**

The training problem can be formulated in terms of _likelihood maximization_

$$\begin{align}
\mathop{\arg\max}_{\mu, \Sigma, \tau}\ & \mathbb{E}_{x \sim X}\left[ L(x, \mu, \Sigma, \tau) \right] \\
\text{s.t. } & \sum_{k=1}^n \tau_k = 1
\end{align}$$

* As usual, the likelihood function $L$ measures how likely it is...
* ...that the training sample $\hat{x}$ is generated by a GMM with parameters $\mu, \Sigma, \tau$

**There's more than one issue here**
...And the first one is dealing with the expectation
**We can approximate the expectation by using the training set**
$$
\mathbb{E}_{x \sim X} \left[ L(x, \mu, \Sigma, \tau)\right] \simeq \prod_{i = 1}^m g(x_i, \mu, \Sigma, \tau)
$$
Technically, this is just an example of Monte-Carlo estimation
* When used for the likelihood of the training data
* ...This is often called "Empirical Risk Minimization" principle

**Let's put everything together**
$$\begin{align}
\mathop{\arg\max}_{\mu, \Sigma, \tau}\ & \prod_{i = 1}^m \sum_{k = 1}^{n} \tau_k f(x, \mu_k, \Sigma_k) \\
\text{s.t. } & \sum_{k=1}^n \tau_k = 1
\end{align}$$
From an optimization point of view, this is _very annoying problem_:
* There's a _constraint_
* There's both _a product and a sum_
* The product cannot be decomposed ($\mu, \Sigma, \tau$ appear in every term)
**So we'll need to get clever!**

#### An Apparent Overcomplication

**We get clever by apparently overcomplicating the problem**

In particular, we introduce a random variable $Z_{i}$ _for each example_

* $Z_{i} = k$  iff $i$-th example was drawn from the $k$-th component
* The $Z_{i}$ are _latent_ since we do not know their value
* We focus on _our_ uncertainty, rather than on the uncertainty in the process

**When computin the PDF, we take the values of $Z_i$ for granted:**
$$
\tilde{g}_i(x_i, z_i, \mu, \Sigma, \tau) = \tau_{z_{i}} f({x_i}, \mu_k, \Sigma_k)
$$
* The value $z_i$ is now _an input to $\tilde{g}_i$_
* ...And we can use it as an index to retrieve the correct $\tau_k$
* This alternative PDF is much easier (there is no sum)!
**The drawback is that we have not uncertainty over _both $X$ and $Z$_**
$$
\mathbb{E}_{x \sim X, z \sim Z} \left[ L(x, z, \mu, \Sigma, \tau)\right]
$$
We can deal with $X$ by using the training set as the single sample

**By doing this we obtain:**

$$
\mathbb{E}_{x \sim X, z \sim Z} \left[ L(x, z, \mu, \Sigma, \tau)\right] \simeq
\mathbb{E}_{z \sim Z} \left[ \prod_{i=1}^m \tilde{g}_i(x_i, z_i, \mu, \Sigma, \tau) \right]
$$
* We cannot use the same technique for $Z$
* ...Since we do not have a sample for **Z** values (they are latent, we don't know them. Instead of the **X** that is approximable with the train set)!
**To deal with the EXPECTATION on $Z$, we add _yet another set of variables_**

* The variables represent the (unknown) distribution of the latent $Z_i$ variables
* In particular, $\tilde{\tau}_{i,k}$ corresponds to $P(Z_i = k)$
**With the new variable, we can compute the expectation _in closed form_:**

$$
\mathbb{E}_{\hat{x} \sim X, \hat{z} \sim Z} \left[ L(\hat{x}, \hat{z}, \mu, \Sigma, \tau)\right] \simeq
\prod_{i=1}^m \prod_{k=1}^n \tilde{g}_i(x_i, z_i, \mu, \Sigma, \tau)^{\tilde{\tau}_{i,k}}
$$

* Intuitively, we if we sampled $Z_{i}$
* ...We would generate $\tilde{\tau}_{i,k}$ samples for each component $k$, for that point $x_i$.
* ...So that the corresponding density is multplied by itself $\tilde{\tau}_{i,k}$ times

**The reworked training problem therefore is**
$$\begin{align}
\mathop{\arg\max}_{\mu, \Sigma, \tau, \tilde{\tau}}\ & \prod_{i=1}^m \prod_{k=1}^n \tilde{g}_i(x_i, z_i, \mu, \Sigma, \tau)^{\tilde{\tau}_{i,k}} \\
\text{s.t. } & \sum_{k=1}^n \tau_k = 1 \\
 & \sum_{k=1}^n \tilde{\tau}_{i,k} = 1 \quad\quad \forall i = 1..m
\end{align}$$

* We have even more variables (the $\tilde{\tau}_{i,k}$ ones)
* ...But they are statistically related! Each $Z_i$ is drawn from $Z$
* ...And there's no longer a combination of sums and products

**We can now use the [Expectation-Maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)**

The EM algorithm is an optimization method based on _alternating steps_

* In the _expectation_ step:
  - We consider $\mu, \Sigma, \tau$ as fixed and we optimize over $\tilde{\tau}$
  - ...I.e. we try to estimate how sampling went
  - After this, we compute the expectation over $Z$ (in a symbolic form)
* In the _maximization_ step:
  - We use the (symbolic) expectation over $Z$ from before
  - We consider $\tilde{\tau}$ as fixed and we optimize over $\mu, \Sigma, \tau$
  
**The method stops when likelihood improvement become too small**
It can be proved to converge to a _local optimum_ under reasonable assumptions

1. **Let's see the _EXPECTATION STEP_ in our case**

	* This is where we handle _the latent variables_
	* We consider $\mu, \Sigma, \tau$ fixed, so that we need to solve:
	
	$$\begin{align}
	\mathop{\arg\max}_{\tilde{\tau}}\ & \prod_{i=1}^m \prod_{k=1}^n \tilde{g}_i(x_i, z_i, \mu, \Sigma, \tau)^{\tilde{\tau}_{i,k}} \\
	\text{s.t. } & \sum_{k=1}^n \tilde{\tau}_{i,k} = 1 & \forall i = 1..m
	\end{align}$$
	
	* The constraint on $\tau$ is no longer needed (it's always satisfied) 
	* The optimization problem can be _easily decomposed_
	* ...So we can optimize _over each example individually_
	
	**Since $\tau$ is fixed, the expectation over $Z$ can be computed exactly**
	
	By substituting $\tilde{g}_i$, _for a single example $i$_ we have:
	
	$$\begin{align}
	\mathop{\arg\max}_{\tilde{\tau}}\ & \prod_{k=1}^n \left(\tau_{k} f({x_i}, \mu_k, \Sigma_k)\right)^{\tilde{\tau}_{i,k}}  \\
	\text{s.t. } & \sum_{k=1}^n \tilde{\tau}_{i,k} = 1
	\end{align}$$
	
	Which (since $\mu, \Sigma, \tau$ are fixed) is solved by choosing:
`
$$
\tilde{\tau}_{i,k} = \frac{\tau_{k} f({x}_i, \mu_k, \Sigma_k)}{\sum_{h=1}^n \tau_{h} f({x}_i, \mu_k, \Sigma_k)} 
$$
2. **For the _MAXIMIZATION_ step we provide only the main ideas**
	Each $\tau_k$ is estimated as the relative sum of the corresponding $\tilde{\tau}_{i,k}$ variables:
	$$
	\tau_{k} = \frac{1}{m} \sum_{i = 1}^m \tilde{\tau}_{i,k}
	$$
	The $\mu_k$ and $\Sigma_k$ parameters can be estimated based on classical methods
	* In particular, we give to each example a _sample weight equal to $\tilde{\tau}_{i,k}$_
	* Then we estimate $\mu$ and $\Sigma$ via a Least Square approach
### GMM in ACTION
```python
gm = GaussianMixture(n_components=2, random_state=4)
gm.fit(train_x);
```
**We need to specify the number of components a priori**
* We can tune it using a **maximum likelihood approach** on a validation set (MLE on test set can work for different models, because the principle is the same. I have a PDF estimator and I multiply the resulting probability related to each point of validation to get a result. I do gridsearch to get the highest result)
* ...Or using other criteria (e.g. elbow method) 

*NOTE* before everything, we STANDARDIZE THE DATA, even if it is not required in GMMs , as they have enough expressive power to learn whatever proba dist there is. 
The reason is that
*  I want that the initial guess of the learnable parameters of GMM is GOOD. Skitlearn assign to  $\mu$ and $\sum$  values close to  0. But if the mean of a component is 130, the algorithm will converge much slower. If instead all the values are **already** close to 0 then skitlearn will converge much faster.
```python
opt = GridSearchCV(GaussianMixture(), {'n_components': [2, 4, 8]}, cv=5)
opt.fit(trdata[inputs])
print(f'Best parameters: {opt.best_params_}')
```
*Best parameters: {'n_components': 4}*
* While training is slow
* ...Generating the alarm signal is now much faster
```python
ldens = opt.score_samples(hpcs[inputs])
signal = pd.Series(index=hpcs.index, data=-ldens)
```
I will omit the alarm signal because it is similar to that of KDE.
We do threshold optimization as usual with the cost that is defined in the above section:
```
c_alarm, c_missed, tolerance = 1, 5, 12
cmodel =  util.HPCMetrics(c_alarm, c_missed, tolerance)

th_range = np.linspace(1e4, 1e9, 1000)
th, val_cost = util.opt_threshold(signal[tr_end:val_end],
                                        valdata['anomaly'],
                                        th_range, cmodel)
print(f'Best threshold: {th:.3f}')
tr_cost = cmodel.cost(signal[:tr_end], hpcs['anomaly'][:tr_end], th)
print(f'Cost on the training set: {tr_cost}')
print(f'Cost on the validation set: {val_cost}')
ts_cost = cmodel.cost(signal[val_end:], hpcs['anomaly'][val_end:], th)
print(f'Cost on the test set: {ts_cost}')
```
*Best threshold: 862864234.234
Cost on the training set: 0
Cost on the validation set: 242
Cost on the test set: 275*
The results are also similar to those from KDE

**Finally, we can have a look at how the model is using its components**
```
zvals = opt.predict(hpcs[inputs])
zsignal = pd.Series(index=hpcs.index, data=zvals)
util.plot_dataframe(hpcsv[inputs], zsignal, figsize=figsize)
```
opt.predict maps each element of the data into the component that  most probably (given the learnt parameters) generated that element.
* The results may vary, since some steps of the process are stochastic
* ...But **typically one or more component will be use for a single, long job**, and this makes sense, because feature values inside a single job are more or less similar (and hence sampled from the same learnt component).![[componentsGMMs.png]]
In THIS photo infact we have this behaviour.

## AUTOENCODER APPROACH

**Autoencoders can be used for anomaly detection**
...By using the _reconstruction error as an anomaly signal_, e.g.:
$$
\left\|x - \mathit{d}(\mathit{e}(x, \theta_e), \theta_d)\right\|_2^2 \geq \epsilon
$$
**This approach has some PROs and CONs compared to KDE**

* The _size of a Neural Network_ does not depend on the size of the training set
* Neural Networks have good _support for high dimensional data_
* ...Plus _limited overfitting_ and _fast prediction/detection time_
* However, input reconstruction can be _harder than density estimation_

**Normalization is important for NNs, due to the use of _gradient descent_**

The performance of SGD depends a lot on its starting point

* DL libraries all come with robust weight initialization procedures
  - ...And robust default parameters for the gradient descent algorithms
* ...But those are designed for data that is:
  - Reasonably _close to zero_
  - Mostly _contained in a $[-1, 1]^n$ box

**You _can_ use NNs with non standardize data**
...But expect _far less reliable_ results
* In addition, vector output should _always_ be standardized/normalized
* We'll see why in a short while
### Data Preparation 
```python
#standardize
tr_end, val_end = 3000, 4500
hpcs = hpc.copy()
tmp = hpcs.iloc[:tr_end]
hpcs[inputs] = (hpcs[inputs] - tmp[inputs].mean()) / tmp[inputs].std()
#divide in splits
trdata = hpcs.iloc[:tr_end]
valdata = hpcs.iloc[tr_end:val_end]
tsdata = hpcs.iloc[val_end:]
```
### Building an Autoencoder

**The we can build an autoencoder (we'll use tensorflow 2.0 and keras)**

First, we build the model using (e.g.) the functional API
```python
from tensorflow import keras
from tensorflow.keras import layers, callbacks

input_shape = (len(inputs), )
ae_x = keras.Input(shape=input_shape, dtype='float32')
ae_z = layers.Dense(64, activation='relu')(ae_x)
ae_y = layers.Dense(len(inputs), activation='linear')(ae_z)
ae = keras.Model(ae_x, ae_y)
ae.compile(optimizer='Adam', loss='mse')
cb = [callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
history = ae.fit(trdata[inputs], trdata[inputs], validation_split=0.1,
                 callbacks=cb, batch_size=32, epochs=30, verbose=0)
```

## Obtaining alarm signal
```python
#- These are the reconstructed values for all the input features
preds = pd.DataFrame(index=hpcs.index, columns=inputs, data=ae.predict(hpcs[inputs], verbose=0))
labels = pd.Series(index=hpcs.index, data=(hpcs['anomaly'] != 0), dtype=int)
sse = np.sum(np.square(preds - hpcs[inputs]), axis=1)
# each element sse_i in sse is the SUM of the MSE on each feature of element x_i
signal_ae = pd.Series(index=hpcs.index, data=sse)
util.plot_signal(signal_ae, labels, figsize=figsize)
```

ALSO in this case the result is SIMILAR to the ones of KDE and GMM. WHY????
Im going to explain it RN.

When we train an autoencoder (renamed here as $h$), we solve:

$$
\mathop{\arg\min}_{\theta} \| h(x_i, \theta) - x_i\|_2^2
$$

By expanding the L2 norm, we get:

$$
\mathop{\arg\min}_{\theta} \sum_{i=1}^m \sum_{j=1}^n \left(h_j(x_i, \theta) - x_{i,j}\right)^2
$$

By introducing a $\log$ and $\exp$ transformation we obtain:

$$
\mathop{\arg\min}_{\theta} \log \exp\left(\sum_{i=1}^m \sum_{j=1}^n \left(h_j(x_i, \theta) - x_{i,j}\right)^2\right)
$$From this, we rewriting the outer sum using properties of exponentials:

$$
\mathop{\arg\min}_{\theta} \log \prod_{i=1}^m \exp\left(\sum_{j=1}^n \left(h_j(x_i, \theta) - x_{i,j}\right)^2\right)
$$
Then we rewrite the inner sum in matrix form:

$$
\mathop{\arg\min}_{\theta} \log \prod_{i=1}^m \exp\left(\left(h(x_i, \theta) - x_{i,j}\right)^T I \left(h(x_i, \theta) - x_{i,j}\right) \right)
$$
From this, we make a few adjustment that do not change the optimal solution:

* We negate the argument of $\exp$ and swap the $\arg\min$ for a $\arg\max$
* We multiply exponent by $1/2 \sigma$ (for some constant $\sigma$)
* We multiply the exponential by $1/\sqrt{2\pi}\sigma$

$$
\mathop{\arg\max}_{\theta} \log \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{1}{2}\left(h(x_i, \theta) - x_{i,j}\right)^T (\sigma I) \left(h(x_i, \theta) - x_{i,j}\right) \right)
$$
The term inside the product is the PDF of a multivariate normal distribution

$$
\mathop{\arg\max}_{\theta} \log \prod_{i=1}^m f\left(x_i, h(x_i), \sigma I\right)
$$

* In particular a distribution _centered on $h(x_i)$_
* ...With _independent Normal components_
* ...All having _uniform variance_

**Let's discuss some implications**
When we use a MSE loss, we are _training for maximum likelihood_

* ...Just like density estimators!
* This is actually true for _many_ ML approaches

The _output_ of a (MSE trained) regressor has a _probabilistic interpretation_
* Specifically, the output is the _mean of a conditional distribution_
* The distribution represents the _variability of the target_
* ...Once the effect of the input is taken into account
* Another way to think of it: **_noise_ around the prediction**

1. We are implicitly assuming that the _noise is **normally distributed_**
	* This true in many cases, but **NOT ALWAYS**
	* Solitamente e' vera questa assunzione perche' per il **Central limit theroem** se **SOMMO** random variables che seguono distribuzioni diverse allora ottengo una Gaussian Distribution (e e' letteralmente  il nostro caso perche'  ho $$-\frac{1}{2}\left(h(x_i, \theta) - x_{i,j}\right)^T (\sigma I) \left(h(x_i, \theta) - x_{i,j}\right)  $$
1. We are also assuming that  all output components _have **the same variance_** (that all the features noise distributions centered on $h(x_i)$ for each  $x_i$, have the same variance. Questo si puo vedere dal fatto che si applica la stessa $\sigma$ a tutte le features praticamente). P.S. : E' anche legato AL SINGOLO COMPONENTE DA cio' che ho visto nei CAPITOLI SUCCESSIVI. Quindi tipo se una SINGOLA feature presenta una VARIANCE di un certo tipo nella prima parte del dataset, e poi questa variance va a DIMINUIRE o cambia molto, ALLORA STO VIOLANDO QUESTA ASSUNZIONE.
	* This is WHY WE NEED TO **STANDARDIZE** THE DATA before using MSE loss.
  3. We are also assuming that the **_noise on all output features is independent_** (ovvero che il risultato di un noise non sia legato a quello di un altro noise in un'altra feature. Questo e' espresso dalla presenza della DIAGONAL MATRIX I).
	* This might be true even if the output features themselves are correlated
	* ...But still it is **NOT TRUE** in all cases

All these implicit assumption can make the problem **_harder_** because It is not always true that they hold.
* This is why error reconstruction can be **harder** than density estimation.
* So It may happen that it doesn't work with MSE because one of the hypothesis are violeted, hence we are no longer doing MLE.  If we directly do MLE we are safer  (I have MLE with 0 assumptions).

## Threshold optimization
As always, and the result is
*Best threshold: 197135.678
Cost on the training set: 0
Cost on the validation set: 262
Cost on the test set: 265*

## Multiple signal analysis 
On the other hand, MSE is GOOD because:
* We can analyze the reconstruction error vector to identify the feature that contributed the most to the high reconstruction error, indicating the attribute most responsible for the elevated alarm signal.
```
se = np.square(preds - hpcs[inputs])
signals_ae = pd.DataFrame(index=hpcs.index, columns=inputs, data=se)
util.plot_dataframe(signals_ae, labels, vmin=-5e4, vmax=5e4, figsize=figsize)
```
![[pred.png]]
```
mode_1 = hpcs.index[hpcs['anomaly'] != 0]
# for each anomalous signal, compute the mean value for each feature and sort
tmp =  se.iloc[mode_1].mean().sort_values(ascending=False)
util.plot_bars(tmp, tick_gap=20, figsize=figsize)
```
![[recostructionerror.png]]
From the image we can say that  Errors are concentrated on 10-20 features. In particular on feature *ips_p0_14, ips_p0_10, 12, 11*
* The largest errors are on "ips", then on "util" (utilization)
* This kind of information can be _very valuable_ for a domain expert!