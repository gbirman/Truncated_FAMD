
# Truncated_FAMD

`Truncated_FAMD` is a library for prcessing [factor analysis of mixed data](https://www.wikiwand.com/en/Factor_analysis). This includes a variety of methods including [principal component analysis (PCA)](https://www.wikiwand.com/en/Principal_component_analysis) and [multiply correspondence analysis (MCA)](https://www.researchgate.net/publication/239542271_Multiple_Correspondence_Analysis). The goal is to provide an efficient and truncated implementation for each algorithm along with a scikit-learn API.

## Table of contents

- [Usage](##Usage)
  - [Guidelines](###Guidelines)
  - [Principal component analysis (PCA)](#principal-component-analysis-pca)
  - [Correspondence analysis (CA)](#correspondence-analysis-ca)
  - [Multiple correspondence analysis (MCA)](#multiple-correspondence-analysis-mca)
  - [Multiple factor analysis (MFA)](#multiple-factor-analysis-mfa)
  - [Factor analysis of mixed data (FAMD)](#factor-analysis-of-mixed-data-famd)
- [Going faster](#going-faster)




`Truncated_FAMD` doesn't have any extra dependencies apart from the usual suspects (`sklearn`, `pandas`, `numpy`) which are included with Anaconda.

## Usage

```python
import numpy as np; np.random.set_state(42)  # This is for doctests reproducibility
```

### Guidelines
`Truncated_FAMD` integrates the power of automatic selection of `svd_solver` according to structure of data and to `n_components` parameter the [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) possesses and capacity of processing sparse input that [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD) has and the support for processing data in a minibatch form,making it possible to processing big data,from [Incremental PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA).Therefore,`Truncated_FAMD` is appropriate for processing big sparse input.

Each base estimator(CA,PCA) provided by `Truncated_FAMD` extends scikit-learn's `(TransformerMixin,BaseEstimator)`.which means we could use directly `fit_transform`,and `(set_params,get_params)` methods.
 
Under the hood `Truncated_FAMD` uses `partial_fit` method on chunks of data fetched sequentially from the original data,only stores estimates of component and noise variances, in order update `explained_variance_ratio_` incrementally. This is why memory usage depends on the number of samples per batch, rather than the number of samples to be processed in the dataset.


In this package,inheritance relationship as shown  below(A->B:A is superclass of B):

- _BasePCA ->PCA ->CA -> MCA 
- _BasePCA ->PCA ->MFA -> FAMD

You are supposed to use each method depending on your situation:

- All your variables are numeric: use principal component analysis (`PCA`)
- You have a contingency table: use correspondence analysis (`CA`)
- You have more than 2 variables and they are all categorical: use multiple correspondence analysis (`MCA`)
- You have groups of categorical **or** numerical variables: use multiple factor analysis (`MFA`)
- You have both categorical and numerical variables: use factor analysis of mixed data (`FAMD`)

The next subsections give an overview of each method along with usage information. The following papers give a good overview of the field of factor analysis if you want to go deeper:

- [A Tutorial on Principal Component Analysis](https://arxiv.org/pdf/1404.1100.pdf)
- [Theory of Correspondence Analysis](http://statmath.wu.ac.at/courses/CAandRelMeth/caipA.pdf)
- [Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions](https://arxiv.org/pdf/0909.4061.pdf)
- [Computation of Multiple Correspondence Analysis, with code in R](https://core.ac.uk/download/pdf/6591520.pdf)
- [Singular Value Decomposition Tutorial](https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf)
- [Multiple Factor Analysis](https://www.utdallas.edu/~herve/Abdi-MFA2007-pretty.pdf)




###	Principal-Component-Analysis: PCA

**PCA**(standard_scaler=True,n_components=2,svd_solver='auto',whiten=False,copy=True,
                 tol=None,iterated_power=2,batch_size =None,random_state=None):
	
**Args:**
- `standard_scaler` (bool): Whether to standard_scaler each column's  or not.
- `n_components` (int, float, None or string): The number of principal components to compute.See [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
- `svd_solver`(string {`auto`, `full`, `arpack`, `randomized`}):Noter that if input data is sparse,`svd_solver`:{`arpack`,`randomized`}.See [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
- `iterated_power` (int): The number of iterations used for computing the SVD when `svd_solver` == `randomized`.
- `tol`(float >= 0, optional (default .0)):Tolerance for singular values computed by `svd_solver` == `arpack`.
- `copy` (bool): Whether to perform the computations inplace or not.
- `batch_size`( int or None):The number of samples to use for each batch. Only used when calling fit. If `batch_size` is None, then `batch_size` is inferred from the data and set to 5 * n_features, to provide a balance between approximation accuracy and memory consumption.
- `random_state`(int, RandomState instance or None, optional (default=None):The seed of the -pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
Return ndarray (M,k),M:Number of samples,K:Number of components.

**Examples:**
```
>>>import numpy as np
>>>from Light_Famd import PCA
>>>X = pd.DataFrame(np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]),columns=list('ABC'))
>>>pca = PCA(n_components=2)
>>>pca.fit(X)
PCA(check_input=True, copy=True, engine='auto', n_components=2, n_iter=3,
  random_state=None, rescale_with_mean=True, rescale_with_std=True)

>>>print(pca.explained_variance_)
[2.37837661 0.02162339]

>>>print(pca.explained_variance_ratio_)
[0.99099025 0.00900975]
>>>print(pca.column_correlation(X))  #You could call this method once estimator is >fitted.correlation_ratio is pearson correlation between 2 columns values,
where p-value >=0.05 this similarity is `Nan`.
          0   1
A -0.995485 NaN
B -0.995485 NaN

>>>print(pca.transform(X))
[[ 0.82732684 -0.17267316]
 [ 1.15465367  0.15465367]
 [ 1.98198051 -0.01801949]
 [-0.82732684  0.17267316]
 [-1.15465367 -0.15465367]
 [-1.98198051  0.01801949]]
>>>print(pca.fit_transform(X))
>[[ 0.82732684 -0.17267316]
 [ 1.15465367  0.15465367]
 [ 1.98198051 -0.01801949]
 [-0.82732684  0.17267316]
 [-1.15465367 -0.15465367]
 [-1.98198051  0.01801949]]

```
###	Correspondence-Analysis: CA
CA class inherits from  PCA  class.


**Examples:**
```
>>>import numpy as np
>>>from Light_Famd import CA
>>>X  = pd.DataFrame(data=np.random.randint(0,100,size=(10,4)),columns=list('ABCD'))
>>>ca=CA(n_components=2,iterated_power=2)
>>>ca.fit(X)
CA(batch_size=None, copy=True, iterated_power=2, n_components=2,
  random_state=None, svd_solver='auto', tol=None))

>>> print(ca.explained_variance_)
[0.02811427 0.01346975]

>>>print(ca.explained_variance_ratio_)
[0.54703122 0.26208655]

>>>print(ca.transform(X))
[[-0.74276079 -0.24252589]
 [-0.02821543  0.27099114]
 [ 0.47655683 -0.53616059]
 [ 0.11871109 -0.10247506]
 [ 0.06085895 -0.15267951]
 [ 0.89766224 -0.19222481]
 [ 0.683192    0.67379238]
 [-0.66493196 -0.08886992]
 [-0.81955305 -0.08935231]
 [-0.21371233  0.48649714]]


```

###	Multiple-Correspondence-Analysis: MCA
MCA class inherits from  CA  class.

```
>>>import pandas as pd
>>>X=pd.DataFrame(np.random.choice(list('abcde'),size=(50,4),replace=True),columns =list('ABCD'))
>>>print(X)
   A  B  C  D
0  e  a  a  b
1  b  e  c  a
2  e  b  a  c
3  e  e  b  c
4  b  c  d  d
5  c  d  a  c
6  a  c  e  a
7  d  b  d  b
8  e  a  e  e
9  c  a  e  b
...
>>>mca=MCA(n_components=2,iterated_power=2)
>>>mca.fit(X)
MCA(batch_size=None, copy=True, iterated_power=2, n_components=2,
  random_state=None, svd_solver='auto', tol=None)

>>>print(mca.column_correlation(X))
            0         1
A_a       NaN       NaN
A_b -0.343282 -0.450314
A_c       NaN -0.525714
A_d  0.606039       NaN
A_e -0.482576  0.561833
B_a       NaN -0.303963
B_b  0.622119  0.333704
B_c       NaN       NaN
B_d -0.396896       NaN
B_e       NaN  0.359454
C_a       NaN  0.586749
C_b -0.478460       NaN
C_c       NaN -0.389922
C_d       NaN       NaN
C_e  0.624175       NaN
D_a -0.579454  0.453070
D_b       NaN       NaN
D_c       NaN -0.592007
D_d  0.487821       NaN
D_e       NaN       NaN

>>>print(mca.explained_variance_)
[0.0104482  0.00964206]

>>>print(mca.explained_variance_ratio_)
[0.00261205 0.00241051]

>>>print(mca.transform(X)) 
[[ 4.75897353e-01  1.18328365e+00]
 [-7.44935557e-01  8.80467767e-01]
 [ 8.75427551e-01  5.25160608e-01]
 [ 4.59454326e-01 -4.06521487e-01]
 [ 9.37769179e-01 -7.65735918e-01]
 [-8.34480014e-01  9.82195557e-01]
 [-4.01418791e-03 -9.82014024e-01]
 [-9.98029713e-02  5.25646968e-01]
 [-4.70148309e-01  1.71969029e-03]
 [-8.88880685e-01 -3.95681877e-01]
 [ 1.73157292e+00  3.59962430e-01]
 [ 5.56384642e-01 -4.90593710e-01]
 [-8.34480014e-01  9.82195557e-01]
 [-4.66163214e-01 -1.04999945e+00]
 [-3.65088651e-01  5.85105538e-02]
 [ 1.02856977e+00 -5.33364595e-01]
 [-4.94864281e-01 -7.14484346e-01]
 [-5.47243985e-01  2.59249764e-01]
 [-1.20025145e-02  4.19830209e-01]
 [ 8.96709363e-01  1.29732542e-01]
 [-2.44747616e-01 -5.78512715e-01]
...

```
###	Multiple-Factor-Analysis: MFA
MFA class inherits from  PCA  class.
Since FAMD class inherits from  MFA and the only thing to do for FAMD is to determine `groups` parameter compare to its  superclass `MFA`.therefore we skip this chapiter and go directly to `FAMD`.


###	Factor-Analysis-of-Mixed-Data: FAMD
The `FAMD` inherits from the `MFA` class, which entails that you have access to all it's methods and properties of `MFA` class.
```
>>>import pandas as pd
>>>X_n = pd.DataFrame(data=np.random.randint(0,100,size=(100,2)),columns=list('AB'))
>>>X_c =pd.DataFrame(np.random.choice(list('abcde'),size=(100,4),replace=True),columns =list('CDEF'))
>>>X=pd.concat([X_n,X_c],axis=1)
>>>print(X)
A   B  C  D  E  F
0   32  26  e  b  b  c
1   41  90  e  c  c  e
2   16   2  a  b  b  c
3   22  74  a  d  d  a
4   97  41  d  b  b  a
5   35  18  c  d  a  a
6   95  16  c  d  a  a
7   47   1  c  e  e  c
8    2  24  c  b  c  e
9   82  95  b  c  a  a
10  93  60  a  d  b  e
11  36  56  e  c  a  a
12  30  75  b  c  b  e
13  20  68  b  e  e  a
14  94  98  b  c  e  c
15   8  87  c  b  c  c
16  34  35  c  a  c  b
17  56   6  c  a  d  d
18  33  94  e  a  c  d
19  76  42  a  c  b  c
20  83  62  a  e  d  c
21  65  63  d  e  d  d
22   4  12  a  d  a  a
23  73  38  a  e  e  b
...


>>>famd = FAMD(n_components=2)
>>>famd.fit(X)
FAMD(batch_size=None, copy=True, iterated_power=2, n_components=2,
   random_state=None, svd_solver='auto', tol=None)

>>>print(famd.explained_variance_)
[2.09216762 1.82830854]

>>>print(famd.explained_variance_ratio_)
[0.19719544 0.17232563]

>>> print(famd.column_correlation(X))
            0         1
C_a       NaN -0.545526
C_b       NaN  0.329485
C_c       NaN       NaN
C_d       NaN       NaN
C_e  0.233212  0.430677
D_a  0.308279       NaN
D_b       NaN       NaN
D_c       NaN  0.549633
D_d  0.331463 -0.364919
D_e -0.538894 -0.215123
E_a  0.403095 -0.468114
E_b       NaN  0.203875
E_c  0.199005  0.402092
E_d -0.241785       NaN
E_e -0.278375       NaN
F_a  0.432976 -0.237807
F_b  0.210518       NaN
F_c -0.820450       NaN
F_d       NaN       NaN
F_e  0.240326  0.480436


>>>print(famd.transform(X)) 
[[-10.63120157   9.03256199]
 [  7.2594962   20.41834095]
 [-15.25371982  -2.37627147]
 [  3.11348646 -14.51376937]
 [  1.16397889   2.77352044]
 [ 14.62345611 -14.83857274]
 [ 14.62345611 -14.83857274]
 [-17.26519048  -6.58745196]
 [  4.62106121   9.22232575]
 [  9.60947121   2.49750339]
 [  2.14508201  -4.2707566 ]
 [ 12.26245577   3.06721231]
 [  1.2800622   17.53643902]
 [ -4.38821966  -2.6828045 ]
 [-12.4878278    8.57298192]
...

>>>print(famd.fit_transform(X))
[[-10.63120157   9.03256199]
 [  7.2594962   20.41834095]
 [-15.25371982  -2.37627147]
 [  3.11348646 -14.51376937]
 [  1.16397889   2.77352044]
 [ 14.62345611 -14.83857274]
 [ 14.62345611 -14.83857274]
 [-17.26519048  -6.58745196]
 [  4.62106121   9.22232575]
 [  9.60947121   2.49750339]
 [  2.14508201  -4.2707566 ]
 [ 12.26245577   3.06721231]
 [  1.2800622   17.53643902]
 [ -4.38821966  -2.6828045 ]
 [-12.4878278    8.57298192]
...

```







