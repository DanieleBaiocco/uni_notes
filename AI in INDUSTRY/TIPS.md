###  Occam's Razor
**In practice, it's often a good idea to _start with a simple approach_**

- If it works well, then you have _a solution_
- If it does not, they you have _a baseline_

In both cases, you win!

### Concatenation in Pandas
Always do concatenations in a single step in `pandas`**

It's way faster than appending `DataFrame` objects one by **one**

### GMMs applications
GMMS have strong PROS:
1. It can reconstruct/approximate probability distributions that are SMOOTH ENOUGH, so its expressive power is HIGH (the majority of proba.dist are smooth enough).
2. The computational cost is reasonable: train is fast, prediction is even faster (in prediction we must only loop throught each component, in KDE we had to loop through each training instance).
3. It is also good for interpretability as It gives additional information, along with the prediction of the PDF.
**GMMs are very flexible in terms of what they can do**

1. We can use them to _evaluate the (log) density_ of a sample:
```
pred_lf = np.exp(gm.score_samples(train_x))
print('Log densities:', pred_lf)
```
2. We can use them to _generate a sample_:
```
pred_x, pred_z = gm.sample(3)
print('Sampled values:', str(pred_x).replace('\n', ' '))
print('Sampled components:', pred_z)
```
OUTPUT: 
*Sampled values: \[\[ 0.18426123  0.89138752]  \[-0.12924724  0.96302585] \[-0.18728814  0.61701288]]
Sampled components: \[1 1 1]*
3.  We can estimate the _probability that a sample belongs to a component_:
```
pred_p = gm.predict_proba(train_x)
print('Probability of belonging to a component:')
print(pred_p[:3])
```
OUTPUT: *Probability of belonging to a component:
\[9.99999974e-01 2.57233358e-08]
 \[1.48175059e-05 9.99985182e-01]
 \[1.13219614e-06 9.99998868e-01]]*
By picking the maximum probability, we can _assign samples to a component_ :
```
pred_c = gm.predict(train_x)
print(pred_c[:3])
```
OUTPUT: *\[0 1 1]*

**GMMs can certainly act as density estimators**
...But can do much more!
* Sampling
* Component assignment
* ...And therefore _clustering_

This is so true that GMM are often presented as a generalization of k-means

**And this (partially) addresses the last limitation of KDE** (that returns ONLY density probabilities and don't give further information about the predictions)

* With GMMs we can obtain additional information in addition to the densities

### FP AND FN explanation with GMMs
I have FP when the model sees instance values that it hasn't already seen before.
GMM just sees whether these points has already been seen in the training set. The answer in the case of FP is NO. For this reason it is crucial that the training set is representative of all the types of configuration of values.
I have FN when the values that yield an anomaly are NOT that different from the *normal* ones that are the ones of the training set in which there are no anomalies. I could also retrain the model If I find points that don't contain an anomaly that are FP, by putting them in the training set and retrain everything

### AUTOENCODERS work IF
Autoencoders work only if input has features that are **CORRELATED**, because It uses these correlations to represent in the latent space a concentrated part of the input and then from there , knowing the correlations to recontruct the input.
## NNs and Standardization

**Normalization is important for NNs, due to the use of _gradient descent_**

The performance of SGD depends a lot on its starting point

* DL libraries all come with robust weight initialization procedures
  - ...And robust default parameters for the gradient descent algorithms
* ...But those are designed for data that is:
  - Reasonably _close to zero_
  - Mostly _contained in a $[-1, 1]^n$ box_
  
**RICORDA: ML MODELS sono FLESSIBILISSIMI, FLESSIBILISSIMI, FLESSIBILISSIMI.
Serve ESPERIENZA in Machine Learning, non ho KNOWN TASKS e KNOWN WAYS of solving the tasks.**