The pdf is this file:///C:/Users/danie/Downloads/Tan-Steinbach-Kumar-DataMining-data_exploration%20(1).pdf.

## Vectors
IMPORTANTE:
The basis vectors are usually orthogonal. The orthogonality of vectors is an extension of the two-dimensional notion of perpendicular lines and will be defined more precisely later on. Conceptually, orthogonal vectors are unrelated or independent. If basis vectors are mutually orthogonal, then expressing a vector as a linear combination of basis vectors effectively decomposes the vector into a number of independent components.
Thus, a vector in an n-dimensional space can be considered to be an ntuple of scalars (numbers).


Hence, we will refer to the components of a vector v by using the notation v = (v1, v2,...,vn−1, vn). With reference to the equation, $$ v = \sum_{i=1}^{n} a_i * u_i $$
we have that $$v_i = a_i $$
## Ortogonality
In Euclidean space it can be shown that the dot product of two (non-zero) vectors is 0 if and only if they are perpendicular. . We say that such vectors are orthogonal. The dot product can also be used to compute the length of a vector

$$ length(u) = \sqrt{u · u} $$
that is called the L1 NORM  || u || of a vector u.

Given a vector u, we can find a vector that is pointing in the same direction as u, but is of unit length, by dividing each component of u by its length; i.e., by computing $$u/||u||$$. We say that we have normalized the vector to have an L2 norm of 1.

Parte successiva non chiarissima:
Given the notation for the norm of a vector, the dot product of a vector can be written as
$$
\mathbf{u} \cdot \mathbf{v} = \lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert \cos(\theta)
$$
where θ is the angle between the two vectors. By grouping terms and reordering, this can be rewritten as
$$
\mathbf{u} \cdot \mathbf{v} = (\lVert \mathbf{v} \rVert \cos(\theta)) \lVert \mathbf{u} \rVert = \mathbf{v_u} \lVert \mathbf{u} \rVert
$$
where $$
\mathbf{v_u} = \lVert \mathbf{v} \rVert \cos(\theta)
$$
represents the length of v in the direction of u as illustrated in Figure A.2.
![[Cattura.png]]
Nota che se ho v ortogonale a u allora v_u sara' 0 , e il dot product tra u e v_u sara' 0.

If u is a unit vector, then the dot product is the component of v in the direction of u (just the v_u). We refer to this as the **orthogonal projection** of v onto u.

An important consequence of this is that, given a set of orthogonal vectors of norm 1 that form a basis of a vector space, we can find the components of any vector with respect to that basis by taking the dot product of the vector with each basis vector.
Questo perche' come detto sopra basta prendere l'ortoginal projection su una determinata base per capire il valore della coordinata su quella base li' (pensa a quando trovi i valori di un punto su un piano cartesiano, prima fai ortogonal projection sull'asse x e poi sull'asse y, che altro non sarebbe che il dot product con gli unit vectors \[0,1\] e \[1, 0]\).

A concept that is closely related to that of orthogonality is the notion of linear independence

(Linear Independence). A set of vectors is linearly independent if no vector in the set can be written as a linear combination of the other vectors in another set.

If a set of vectors is not linearly independent, then they are linearly dependent. Note that we want our basis to consist of a set of vectors such that no vector is linearly dependent with respect to the remaining basis vectors, because if this were so, then we could eliminate that vector and still have a set of vectors that span the entire vector space.

If we choose our basis vectors to be mutually orthogonal (independent), then we automatically obtain a linearly independent set since any two vectors that are orthogonal are linearly independent.


## Data as Vectors 
Once we have represented our data objects as vectors, we can perform various operations on the data that derive from a vector viewpoint. For example, using various vector operations, we can compute the similarity or distance of two vectors. In particular, the cosine similarity of two vectors is defined as
![[data_as_vector.png]]
This similarity measure **does not take into account the magnitude (length) of the vectors** (infatti e' normalizzato), but is only concerned with the degree to which two vectors point in the same direction.

In terms of documents, this means that two documents are the same if they contain the same terms in the same proportion. Terms that do not appear in both documents play no role in computing similarity.

We can also simply define the distance between two vectors (points). If u and v are vectors, then the Euclidean distance between the two vectors (points) is simply:

![[euclidian_dist.png]]

Also, for vector data, it is meaningful to compute the mean of the set of vectors, which is accomplished by computing the mean of each component.
Some clustering approaches, such as K-means (Chapter 7) work by dividing the data objects into groups (clusters) and characterizing each cluster by the mean of the data objects (data vectors).


The idea is that a good cluster is one in which the data objects in the cluster are close to the mean, where closeness is measured by Euclidean distance for data like the Iris data and by cosine similarity for data like document data.


## Matrices
If we have an n by 1 column vector u, then we can view the multiplication of an m by n matrix A by this vector on the right as a transformation of u into an m-dimensional column vector v = Au.

Similarly, if we multiply A by a (row) vector u = \[u1,...,um\] on the left, then we can view this as a transformation of u into an n-dimensional row vector v = uA

**Thus, we can view any m by n matrix A as a function that maps one vector space onto another**

In many cases, the transformation (matrix) can be described in easily understood terms.
1. A **scaling matrix** leaves the direction of the vector unchanged, but changes its length. This is equivalent to multiplying by a matrix that is the identity matrix multiplied by a scalar.

2. A **rotation matrix** changes the direction of a vector, but leaves the magnitude of the vector unchanged. This amounts to a change of coordinate system. 

3. A **reflection matrix** reflects a vector across one or more coordinate axes. This would be equivalent to multiplying some of the entries of the vector by −1, while leaving the other entries unchanged. 

4. A **projection matrix** takes vectors into a lower dimensional subspace. The simplest example is the modified identity matrix where one or more of the 1’s on the diagonal have been changed into 0’s. Such a matrix eliminates the vector components corresponding to those zero entries, while preserving all others

Of course, a single matrix can do two kinds of transformations at once, e.g., scaling and rotation

# Dimensionality Reduction
PCA.
PCA has several appealing characteristics. First, it tends to identify the strongest patterns in the data. Hence, PCA can be used as a pattern-finding technique. Second, often most of the variability of the data can be captured by a small fraction of the total set of dimensions. As a result, dimensionality reduction using PCA can result in relatively low-dimensional data and it may be possible to apply techniques that don’t work well with high-dimensional data. Third, since the noise in the data is (hopefully) weaker than the patterns, dimensionality reduction can eliminate much of the noise. This is beneficial both for data mining and other data analysis algorithms.

Statisticians summarize the variability of a collection of multivariate data; i.e., data that has multiple continuous attributes, by computing the covariance matrix S of the data. 
**Definition B.1**.** Given an m by n data matrix D, whose m rows are data objects and whose n columns are attributes, the covariance matrix of D is the matrix S, which has entries sij defined as
$$ sij = covariance(d_{∗i}, d_{∗j} ).$$
The covariance of two attributes is defined in Appendix C, and is a measure of how strongly the attributes vary together.
If i = j, i.e., the attributes are the same, then the covariance is the variance of the attribute.

A goal of PCA is to find a transformation of the data that satisfies the following properties::
1. Each pair of new attributes has 0 covariance (for distinct attributes)
2. The attributes are ordered with respect to how much of the variance of the data each attribute captures.
3. The first attribute captures as much of the variance of the data as possible.
4. Subject to the orthogonality requirement, each successive attribute captures as much of the remaining variance as possible.
A transformation of the data that has these properties can be obtained by using **eigenvalue analysis** of the covariance matrix.
Let λ1,...,λm be the eigenvalues of S. The eigenvalues are all non-negative and can be ordered such that $$λ_1 ≥ λ_2 ≥ ...λ_m−1 ≥ λ_m$$
(Covariance matrices are examples of what are called positive semidefinite matrices, which, among other properties, have non-negative eigenvalues.)

# Probability and Statistics