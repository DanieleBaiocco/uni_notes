# DENSITY ESTIMATION APPROACH

We can characterize a continuous distribution via its _density_
* Given a random variable $X$ with values $x$
* ...We care about its _Probability Density Function_ $f(x)$
**Since anomalies are assumed to be unlikely**
...Our detection condition can be stated as:
$$
f(x) \leq \varepsilon
$$
* Where $\varepsilon$ is a (scalar) threshold

**We need NOW one way to _estimate probability densities_** 
For some random process with n-dimensional variable $x$:

* Given the true density function $f(x): \mathbb{R}^n \rightarrow \mathbb{R}^+$
* ...And a second function $\hat{f}(x, \theta)$ with the same input, and parameters $\theta$

We want to _make the two as similar as possible_
Given some suitable loss function $L(y, \hat{y})$, we would need to solve:

$$
\text{argmin}_{\theta} L(\hat{f}(x, \theta), f(x))
$$
* where $x$ represents the training data
But **NOTE** that **we dont have the true probability density function**. So we cannot proceed in the usual **supervised learning** way.

**Density estimation is an _unsupervised_ learning problem**

**It can be solved via a number of techniques:**

* Simple histograms
* Kernel Density Estimation
* Gaussian Mixture Models
* [Normalizing Flows](https://arxiv.org/abs/1505.05770)
* [Non Volume Preserving (NVP) transformations](https://arxiv.org/abs/1605.08803)

## Simple histograms
We make an histogram of the number of calls (how many times a specific number of calls happens in the entire dataset?). Then when we have new number of calls values we can compute the PROBABILITY of that number of calls (NOTE: it doesn't take into account what happened previously, it just gives a probability given a value). If it is LOW (that number appears in the dataset only a few times), then we can say that that number of calls is an ANOMALOUS number of calls value.![[probadist.png]]

## Kernel Density Estimation
Instead of the histogram we would like to have something **smoother**, that approximate this **Probability Density Function**.

**In _Kernel Density Estimation (KDE)_, the main idea is that:**
* Wherever (in input space) there is a sample
* ...It's likely that there are more
So, we assume that _each training sample is the center for a density "kernel"_

**Formally, suck kernel $K(x, h)$ is just a valid PDF:**
* $x$ is the input variable (scalar or vector)
* $h$ is a parameter (resp. scalar or matrix) called _bandwidth_
Typical kernels: Gaussian, exponential, cosine, linear...

Kernel density estimation is a **nonparametric model** used for estimating probability distributions. Before diving too deeply into kernel density estimation, it is helpful to understand the concept of **nonparametric estimation**.

### Nonparametric estimation
Unlike traditional estimation techniques, nonparametric estimation does not assume that data is drawn from a known distribution. Rather, nonparametric models determine the model structure from the underlying data. A few examples of nonparametric data are:
1. Data without a strong theoretical link to a known distribution.
2. Data with anomalies such as outliers, shifts, or heavy-tails.
Note that also histograms are nonparametric estimators

### KDE explained
Reference to undestand KDE deeper:
1. https://www.aptech.com/blog/the-fundamentals-of-kernel-density-estimation/
2. https://towardsdatascience.com/kernel-density-estimation-explained-step-by-step-7cc5b5bc4517
In summary: 
To get a sense of the data distribution, we draw probability density functions (PDF). We are pleased when data fit well to a common density function, such as normal, Poisson, geometrical, etc. Then, the **maximum likelihood approach** can be used to fit the density function to the data.
Unfortunately, the data distribution is sometimes **too irregular** and **does not resemble any of the usual PDFs**. In such cases, the Kernel Density Estimator (KDE) provides a rational and visually pleasant representation of the data distribution.

The key to understanding KDE is to think of it as **a function made up of building blocks**, similar to how different objects are made up of Lego bricks. The distinctive feature of KDE is that it employs only **one type of brick, known as the kernel** (‘_one brick to rule them all_’). The key property of this brick is the ability to shift and stretch/shrink. **Each datapoint is given a brick, and KDE is the sum of all bricks**.
This is the formula: ![[KDEformula.png]]
where
![[XKDA.png]]
If we want to compute it for a single value we just use an $x_i$ , instead of the whole vector X.
In this way we get how much likely it is to have $x_i$  number of calls.

**KDE models are not trained in the usual sense**
...But they store internally _all the training samples_
* hence the training set is **_part of the model parameters_**
* This is a property common to most kernel models

### Tuning bandwidth
**There is one thing that we need to train, i.e. the bandwidth $h$**
* We will see a general approach later in the course
* ...But in the _univariate_ case we can apply a rule of thumb:
$$h = 0.9 \min\left(\hat{\sigma}, \frac{\mathit{IQR}}{1.34}\right) m^{-\frac{1}{5}}$$
Where $\mathit{IQR}$ is the inter-quartile range

### Anomaly detection with KDE
**We will work with _log probabilities_**
* This is what sklearn does by default
* ...And simplifies some operations
* I.e. products become sums
* 
**Overall, our anomaly detection condition becomes:**
$$
-\log f(x, \theta) \geq \varepsilon
$$
...Which is equivalent to the previous formulation.

**We will split our data in two segments**
A **_training** set_:
* This will include only data about the **_normal_ behavior**
* **Ideally, there should be no anomalies here** (we do not want to learn them!)
* We will use it to fit a KDE model
A _test set_:
* To assess how well the approach can generalize
**If the training set contains some anomalies**
* **Things may still be fine**, as long as they are very **_infrequent_**
* ...Since we will still learn that they have low probability

The code is the following:
1. We split the data into train and test. We compute the h following the rule of thumb
```
train_end = pd.to_datetime('2014-10-24 00:00:00')
data_tr = data[data.index < train_end]
q1 = data_tr['value'].quantile(0.25)
q3 = data_tr['value'].quantile(0.75)
sigma = data_tr['value'].std()
m =  len(data_tr)
h = 0.9 * min(sigma, (q3-q1)/ 1.34) * m**(-0.2)
```
2. We fit the KDE on the train data
```
kde = KernelDensity(kernel='gaussian', bandwidth=h)
kde.fit(data_tr.values);
```
3. We obtain for each data point $x_i$ (both training and test data) the alarm signal computed as  the probability of having that number of calls $x_i$ negated. In this way rare number of taxi calls will have an higher value then common number of taxi calls.
```
ldens = kde.score_samples(data.values) # Obtain log probabilities
signal = pd.Series(index=data.index, data=-ldens) # Build series with neg. prob.
util.plot_series(signal, labels=labels, windows=windows, figsize=figsize) # Plot
```
The output is:
![[KDEresults.png]]
The inference is way slower than the training, because in training the KernelDensity just store the training points and the bandwidth value. It doesn't do much more. In inference time it actually uses the stored train points, the bandwidth and the data.values points to return PDFs for each data point. 
4. We now detect anomalies, with a random selected threshold:
```
thr = 12
pred = pd.Series(signal.index[signal >= thr])
print(pred.head(), f'#anomalies: {len(pred)}')
```
![[kderesultsPlotted.png]]
- Not very good, but the threshold _is_ random
- There are a many **false positives**, which are ***_very common_** in anomaly detection
### Anomaly detection metric 
**Evaluating the quality of an Anomaly Detection system can be tricky**
* Usually, we do not need to match the anomalies exactly
* Sometimes we wish to anticipate anomalies
* ...But sometimes we just want to detect them in past data

There is no "catch-all" metric, like accuracy in classification
**It is much better to devise a _cost model_**
* We evaluate the cost and benefits of our predictions:
* By doing this, we focus on _the value for our customer_
> **This is important _for all industrial problems!_**
#### A Simple Cost Model
**We will use a simple cost model**
Remember that our goals are:
* Analyzing anomalies
* Anticipating anomalies
First, we define:
* _True Positives_ as windows for which we detect at least one anomaly
* _False Positives_ as detected anomalies that do not fall in any window
* _False negatives_ as anomalies that go undetected
* _Advance_ as the time between an anomaly and when first we detect it
**Then we introduce:**
* A cost $c_{alarm}$ for loosing time in analyzing false positives
* A cost $c_{missed}$ for missing an anomaly
* A cost $c_{late}$ for a late detection (partial loss of value)

Our cost model (simple, but serviceable) is then given by:
```
c_alrm = 1 # Cost of investigating a false alarm
c_missed = 10 # Cost of missing an anomaly
c_late = 5 # Cost for late detection

# A simple cost model
cost = c_alrm * len(fp) + \
       c_missed * len(fn) + \
       c_late * (len([a for a in adv if a.total_seconds() <= 0]))
print(f'The cost with the current predictions is: {cost}')
```
Refactor:
```python
cmodel = util.ADSimpleCostModel(c_alrm, c_missed, c_late)
cost = cmodel.cost(signal, labels, windows, thr)
print(f'The cost with the current predictions is: {cost}')
```
### Finetuning the threshold
We define a Validation set for threshold optimization (note that in this case the validation set is not used to reduce overfitting. The model has already been trained).
Note that this Validation set should contain some anomalies, but some should be left out for testing.
```python
val_end = pd.to_datetime('2014-12-10 00:00:00')
signal_opt = signal[signal.index < val_end]
labels_opt = labels[labels < val_end]
windows_opt = windows[windows['end'] < val_end]
```
We then use this function:
```python
def opt_thr(signal, labels, windows, cmodel, thr_range):
    costs =  [cmodel.cost(signal, labels, windows, thr)
            for thr in thr_range]
    costs = np.array(costs)
    best_idx = np.argmin(costs)
    return thr_range[best_idx], costs[best_idx]
```
So the above code continues in this way:
```
best_thr, best_cost = util.opt_thr(signal_opt, labels_opt, windows_opt,  cmodel, thr_range)
print(f'Best threshold: {best_thr:.3f}, corresponding cost: {best_cost:.3f}')
```
Note that we just used the Training set and the Validation set together. Basically we just splitted the data into TRAIN and TEST (because the validation is used along with the train set).
The result is : *Best threshold: 15.079, corresponding cost: 15.000*
**For all the data (yes, we are cheating a bit) we have:**
```
ctst = cmodel.cost(signal, labels, windows, best_thr)
print(f'Cost on the whole dataset {ctst}')
```
*Cost on the whole dataset 45*
This is **suboptimal**.



# SLIDING WINDOW APPROACH
**Let's have a closer look at our time series**
![[timeseriesplot.png]]
As we can see nearby points tend to have **similar values**, meaning they are _correlated_.
We then want to *determine* the *Correlation interval*.
A useful tool is **Autocorrelation Plots**. They work in the following way
- Consider a range of possible _lags_
- For each lag value 𝑙:
    - Make a copy of the series and shift it by 𝑙 time steps
    - Compute the **Pearson Correlation Coefficient** with the original series
- **Plot the correlation coefficients** over the lag values

**Then we look at the resulting plot:**
- Where the curve is far from zero, there is a significant correlation
- Where it gets close to zero, no significant correlation exists![[autocorrelationplot.png]]
This is the resulting autocorrelation plot. As we can see, there is an high correlation when the Lag values are 4-5. The bigger the Lag values the smaller the Autocorrelation coefficient.

**These correlations are _a source of information_**. They could be exploited to improve our estimated probabilities, but our models so far make _no use_ of them. **How can we take advantage of them?**

#**For example, rather then feeding our model with individual observations** we can use _sequences of observations_ as input. This is a very common approach in time series and in many cases it is a good idea.

## Sliding window idea
**A common approach consist in using a _sliding window_**

![[sliding window.png]]

* We choose a _window length $w$_, i.e. the length of each sub-sequence
* We place the "window" at the beginning of the series
* ...We extract the corresponding observations
* Then, we move the forward by a certain _stride_ and we repeat
 **The result is a table**:
Let $m$ be the number of examples and $w$ be the window length

|               | $\bf s_0$ | $\bf s_1$    | $\bf \ldots$ | $\bf s_{w-1}$ |
| ------------- | --------- | ------------ | ------------ | ------------- | 
| $\bf t_{w-1}$ | $x_0$     | $x_1$        | $\ldots$     | $x_{w-1}$     |
| $\bf t_{w}$   | $x_1$     | $x_2$        | $\ldots$     | $x_{w}$       |
| $\bf t_{w+1}$ | $x_2$     | $x_3$        | $\ldots$     | $x_{w+1}$     |
| $\bf \vdots$  | $\vdots$  | $\vdots$     | $\vdots$     | $\vdots$      |
| $\bf t_{m-1}$ | $x_{m-w}$ | $x_{m-w+1}$  | $\vdots$     | $x_{m-1}$     |

* The first window includes observations from $x_0$ to $x_{w-1}$
* The second from $x_1$ to $x_{w}$ and so on
* $t_i$ is the _time window index_ (where it was applied)
* $s_j$ is the _position_ of an observation _within a window_

**IMPORTANT**: **pandas provides a sliding window _iterator_**

```python
DataFrame.rolling(window, ...)
```
```python
rows = []
for i, w in enumerate(data['value'].rolling(wlen)):
    if i >= wlen-1: rows.append(w.values)
wdata_index = data.index[wlen-1:]
wdata = pd.DataFrame(index=wdata_index, columns=range(wlen), data=rows)
```
**This method works, but _it's a bit slow_**
* We are building our table by rows...
* ...But it is usually _faster to do it by columns_!
* After all, there are usually _fewer columns than rows_
% there is another method but im not going to talk about it cuz I didnt fully undestand it.

## Sequence Input in KDE
There is straightforward approach, using _multivariate_ KDE
* Treat each sequence as a _vector variable_
* Learn an estimator as usual
I can treat windows such that a window **captures all the past history**, because all the local correlation is captured at the  window level). To achieve this we must look at the autocorrelation plot and then choose a fairly high lag value (because we want to capture a lot of the correlation), until it doesnt decrease too much. This ensure the markow property of the windows.

### Bandwidth Choice in Multivariate KDE
First, we need to choose a bandwidth
* We cannot use the (univariate) rule of thumb
* ...But we can use a more general approach
**The basic intuition is that a good bandwidth** will make the actual data register as _more likely_
* Therefore we can pick a _validation set_
* ...And tune the bandwidth for **_maximum likelihood_**
*NOTE*  we assume *statistical independence*, even though it is not true because the windows  **overlap** (stride is equal 1).

**Formally, let $\tilde{x}$ be a _validation_ set of $m$ examples:**
Assuming independent observations, their _estimated probability_ is given by:

$$L(h, x, \bar{x}) = \prod_{i=1}^m \hat{f}(\tilde{x_i}, \bar{x}, h)$$
This is a called a _likelihood function_
* The main input of are the _model parameters_ ($h$ in our case)
* $\hat{f}$ is the density estimator (which outputs a probability)
* $\bar{x}$ the training set
* $\tilde{x}$ is the validation set
So basically for each point in the validation set we compute its probability on the KDE. We then do the product of these probabilities to have how much  $\tilde{x}$ is likely. I want $\tilde{x}$ to be the most likely, by changing the h.
**We can then choose $h$ so as to _maximize the likelihood_ of the validation set given the KDE fitted on the test set**
Meaning that the training problem is given by:

$$\mathop{\arg\max}_{h} \mathbb{E}_{\tilde{x} \sim f(x), \bar{x} \sim f(x)}\left[ L(h, \tilde{x}, \bar{x})\right]$$

* Where $f(x)$ is the true distribution.
I want a validation set that changes, because $\tilde{x}$ must be sampled from the true distribution, hence it must be composed of all the points of the training set (that is the "true distrubution") in rotation. So I must use **cross-validation**. For finding the best h, we will use grid search.
So I have grid search with cross validation.

**As I said**:
* We will approximate $\mathbb{E}$ by sampling multiple $x$ and $\bar{x}$
* ...I.e. multiple validation and training sets
* Then we pick the bandwidth $h^*$ leading to the maximum average likelihood
```python
wdata_tr = wdata[wdata.index < train_end]
params = {'bandwidth': np.linspace(400, 800, 20)}
gs_kde = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv = 5)
gs_kde.fit(wdata_tr)
gs_kde.best_params_
```
**NOTE** We could have done it also in the univariate case. The same procedure could have done there, instead of picking the h bandwith value computed following the rule-of-thumb.

We then plot the signals:
```python
ldens = gs_kde.score_samples(wdata)
signal = pd.Series(index=wdata.index, data=-ldens)
util.plot_series(signal, labels, windows, figsize=figsize)
```
![[signalsWithKDEmultivariate.png]]
### Threshold optimization
```python
signal_opt = signal[signal.index < val_end]
labels_opt = labels[labels < val_end]
windows_opt = windows[windows['end'] < val_end]
thr_range = np.linspace(50, 200, 100)

best_thr, best_cost = util.opt_thr(signal_opt, labels_opt, windows_opt,  cmodel, thr_range)
print(f'Best threshold: {best_thr:.3f}, corresponding cost: {best_cost:.3f}')
```
It is the same as in the univariate case. It is done in the whole train+validation dataset, in the same way.
the result is 
*Best threshold: 104.545, corresponding cost: 7.000*
On the whole dataset is:
```
ctst = cmodel.cost(signal, labels, windows, best_thr)
print(f'Cost on the whole dataset {ctst}')
```
*Cost on the whole dataset 30*.

# TIME DEPENDENT ESTIMATOR APPROACH
## Spotting the Problem
**Let us consider the first two window applications**
* In the first window, the observations are $x_0, x_1$ and so on
* In the second window, the observations are $x_1, x_2$ and so on
$x_0$ is number of taxis as 00:00, $x_1$ at 00:30, and so on
* Hence, the first observation in the first window corresponds to 00:00
* ...But in the second window corresponds to 00:30
**Our estimator learns a distribution for the _observations_:**
* Moving the window forward changes "who is who"
* We learn the distribution of $x_0$ (and its correlations) for multiple $x_0$s, that each time are related to different timesteps!
The learning problem is still well defined, but also _very complex_
**This is the reason for (most of) the noise in the alarm signal**
**Note** that we introduced the windows because we wanted to take advantage of correlation between nearby points.

But if we do the autocorrelation plot with max_lag=96 , we find that the correlation increases and peaks when max_lag  = 48 (precisely after 24 hours). This means that there is a daily pattern in the number of calls. So there is a **period** of 24 hours (a day).

Our sequence-based estimator
* ...Is solving a uselessly complicated problem
* ...And it's not using all the available knowledge
In any problem:
* _Never_ introduce complications unless they are worth it
* _Never_ willingly throw away information
## Time as additional input
**One way to look at that**
...Is that the distribution **_depends on the time of the day_**
* Therefore, we should consider the number of taxi calls $x$
* ...And the time of the day $t$ together
**Let us extract (from the index) the time information information and add it as an additional column:**
```python
dayhour = (data.index.hour + data.index.minute / 60)
data2 = data.copy()
data2['dayhour'] = dayhour
```
**Let us examine the resulting multivariate distribution**:
![[multivariate distributio.png]]
Basically, lets say that for each half-hour we have a distribution over the taxi call values. If the colour is stronger it means that for that half-hour the taxi call values fall for the majority in the same bin.

## Time-dependent estimator in details
**If we feed this information to KDE**
...We learn an estimator for the _joint_ PDF:
$$
f(t, x)
$$

...Which is not exactly what we were looking for
**Assume we flag an anomaly when $f(t, x) \leq \epsilon$**
* This may happen when $x$ (the number of cars) takes an unlikely value
* ...Or when $t$ (the time) does
**Except that the _time is completely predictable_**
* Any different in its estimated density is only due to sampling choices
* In practice, it's a _controlled variable_
**What we really care about is the _conditional density_, i.e.**
$$
f(x \mid t)
$$
* I.e. the density value of the observed value of $x$
* Assuming that the time $t$ is _known_
**Our true anomaly detection conditions should then be:**
$$
f(x \mid t) \leq \varepsilon
$$
...We know how to approximate only to the _joint_ density function $f(t, x)$
**There's more than one way to do it**
...The one we'll see starts with the definition of conditional probability:
$$
f(t, x) = f(x \mid t) f(t)
$$
Meaning that we can detect anomalies by evaluating:
$$
\frac{f(t, x)}{f(t)} \leq \varepsilon
$$
**In order to pull this off, we need**
* An estimator for $f(t, x)$, which we already have
* An estimator for $f(t)$, which we can easily obtain (e.g. using KDE again)
**...But in our specific case, things are _even simpler_**, because **the distribution of time values is uniform**.
**In non-degenerate cases, our condition can always be rewritten:**
$$
\frac{f(t, x)}{f(t)} \leq \varepsilon \quad \longrightarrow \quad f(t, x) \leq \varepsilon f(t)
$$
* But since $f(t)$ is constant this is equivalent to checking the joint probability
* ...With a modified threshold
$$
f(t, x) \leq \varepsilon^\prime
$$
* The threshold $\varepsilon^\prime$ now represents $\varepsilon f(t)$
* ...But since we still need to choose it value, it make little difference to us
**Hence, for this problem we can use $f(t, x)$ for anomaly detection**

## Choosing a Bandwidth 
**We now need to pick a bandwidth**
* We can use grid search and cross-validation again
* ...But this time we need to make sure to **_normalize the data_**
```python
scaler = MinMaxScaler()
data2_n_tr = data2[data2.index < train_end].copy()
data2_n_tr[:] = scaler.fit_transform(data2_n_tr)
data2_n = data2.copy()
data2_n[:] = scaler.transform(data2)
```
We must normalize the data because **this is due to a low-level technical detail:**
* scikit-learn uses a very efficient KDE implementation
* ...But it requires using **_the same bandwidth_** for all input dimensions
So I want that all the input dimensions are of the same scale. (Note that in the window approach each dimension had the same scale because columns were taxi calls, here the second column is the time).
We do MLE with GridSearchCV for optimizing the bandwidth as we already have done in the windows approach.
We then plot the alarm signal genereated by KDE:![[alarmSignalMultivariateKDE.png]]
We optimize the threshold as usual on train+validation set
*Best threshold: 27.273, corresponding cost: 9.000*
And we see the result of the cost on the whole dataset
*Cost on the whole dataset 18* 
**NOTE** there is a second period in the data that is a weekly period. So for example Monday may have less calls than Saturday. This thing is not taken into account in this approach. 

# TIME INDEXED ESTIMATOR APPROACH
**Let's consider a second approach to handle time**
* This consists in _learning many density estimators_:
* Each estimator is _specialized for a given time_ (e.g. 00:00, 00:30, 01:00...)
We can then choose which estimator to use based on the current time

**Formally, what we have is a first _ensemble model_**. It  is an ensemble because it uses different estimators inside, and uses their results to make predictions.
In particular, we obtain our estimated probabilities by evaluating:

$$
f_{g(t)}(x)
$$

* Each $f_i$ function is an estimator
* The $g(t)$ retrieves the correct $f_i$ based (in our case) on the time value
**We'll call this general idea a "selection ensemble"**
In terms of properties:
* Each $f_i$ estimator works with _smaller amounts of data_
* ...But the **individual problems are _easier_**!

**NOTE** in the previous approach, without the information of the sliding windows, I don't take into account the **LOCAL** correlation of the current taxi call value with the values of the previous timesteps. I would like to use both the **GLOBAL** information about time and the **LOCAL** information of what happened a few steps before.

## Training algorithm
**NOTE** the DATASET is turned into **SLIDING WINDOWS**, as in the *SLIDING WINDOWS APPROACH* section.
1. We do a Grid Search to find the optimal bandwidth value for a single estimator of a specific hour . So basically we take a subset of the training set, that contains only the points for hour  04:30:00:
```python
wdata_tr = wdata[wdata.index < train_end]
wdata_tr_test = wdata_tr.iloc[0::48] 
wdata_tr_test.head()
```
![[4.30.mke.png]]
We perform gridsearch with CV, using MLE  (note that MLE is done only on data that refers to this hour).
```python
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': np.linspace(0.01, 0.1, 20)}, cv = 5)
grid.fit(wdata_tr_test)
grid.best_params_
h = grid.best_params_['bandwidth']
```
2.  For sake of simplicity, we'll use the same bandwidth for all estimators, even if we should re-calibrate ℎ for each estimator in principle (this is  a semplification. We should compute different h values, one for each estimator via gridsearch).
3. We then fit a KDE for each hour, with the same bandwidth value learnt in the STEP 1:
```python
day_hours = data_tr.index.hour + data_tr.index.minute / 60
day_hours = day_hours.unique()
kde = {}
for hidx, hour in enumerate(day_hours):
    tmp_data = wdata_tr.iloc[hidx::48]
    kde[hour] = KernelDensity(kernel='gaussian', bandwidth=h)
    kde[hour].fit(tmp_data)
```
4. We generate the signal over all the dataset:
```python
ldens_list = []
for hidx, hour in enumerate(day_hours):
    tmp_data = wdata.iloc[hidx::48]
    tmp_ldens = kde[hour].score_samples(tmp_data)
    tmp_ldens = pd.Series(index=tmp_data.index, data=tmp_ldens)
    ldens_list.append(tmp_ldens)
ldens = pd.concat(ldens_list, axis=0)
ldens = ldens.sort_index()
signal = -ldens
```
We divide the dataset in different subsets. For each, we score the subset, we build a pd.Series and we append it to the signal list. We then concatenate all the array scores (one array score for each timestep), we sort the indexes (in order to get a sequence of timesteps) and we negate the values to have a negative log proba. The resulting signal is this:
![[signalWithKDEplusWindows.png]]
We do threshold optimization as usual:
*Best threshold: 104.04040404040404, corresponding cost: 10*
And we compute the cost on the whole dataset:
*Cost on the whole dataset 10*
**NOTE** we started with 45, then 30, then 18 and now 10. NICE COCK! 
**NOTE** The cost value on the entire dataset is 10: It means that it performs perfectly on the test set (there is no change in the cost value)