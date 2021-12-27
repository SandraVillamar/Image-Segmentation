# Method 3
In Method 3, we will continue trying to classify our cheetah example. Once again we use the decomposition into 8 × 8 image blocks, compute the DCT of each block, and zig-zag scan. We also
continue to assume that the class-conditional densities are multivariate Gaussians of 64 dimensions. The goal is to understand the benefits of a Bayesian solution. For this, using the training data in TrainingSamplesDCT_new_8.mat we created 4 datasets of size given by the table below. They are available in the file TrainingSamplesDCT_subsets_8.mat

![subset_table_screenshot](https://user-images.githubusercontent.com/15370068/147506562-0fe7d541-4ca5-43a4-90a1-67af43ee6524.png)

We start by setting up the Bayesian model. To simplify things a bit we are going to cheat a little. With respect to the class-conditional,
![method3_eqn1](https://user-images.githubusercontent.com/15370068/147506641-a4c3b5dc-aa59-48aa-802d-1715188613ff.png)

we assume that we know the covariance matrix (like Bayes might) by simply replacing it by the sample covariance of the training set, D, that we are working with (and hope he doesn’t notice). That is, we use

![image](https://user-images.githubusercontent.com/15370068/147507544-1d72a1cb-633f-4fda-8446-5924b71c19b4.png)

We are, however, going to assume unknown mean and a Gaussian prior of mean µ<sub>0</sub> and covariance Σ<sub>0</sub>

![image](https://user-images.githubusercontent.com/15370068/147506781-85424ea0-16ce-4c7c-9051-4f0f1454b2e3.png)

Regarding the mean µ<sub>0</sub>, we assume that it is zero for all coefficients other than the first (DC) while for the DC we consider two different strategies:

  - strategy 1 : µ<sub>0</sub> is smaller for the (darker) cheetah class (µ<sub>0</sub> = 1) and larger for the (lighter) grass class (µ<sub>0</sub> = 3).
  - strategy 2 : µ<sub>0</sub> is equal to half the range of amplitudes of the DCT coefficient for both classes (µ<sub>0</sub> = 2);

For the covariance Σ<sub>0</sub> we assume a diagonal matrix with (Σ<sub>0</sub>)<sub>ii</sub> = αw<sub>i</sub>. The mean µ<sub>0</sub> (for the two strategies) and the weights w<sub>i</sub> are given in the files Prior_1.mat (strategy 1) and Prior_2.mat (strategy 2).

A) Consider the training set D<sub>1</sub> and strategy 1 (we will use this strategy until D). For each class, compute the covariance Σ of the class-conditional, and the posterior mean µ<sub>1</sub>, and covariance Σ<sub>1</sub> of

![image](https://user-images.githubusercontent.com/15370068/147507207-65008e62-f881-4a98-8208-c758df07c6f5.png)

Next, compute the parameters of the predictive distribution

![image](https://user-images.githubusercontent.com/15370068/147507228-3c75f2bc-cc3e-46b0-8fea-05b549df2547.png)

for each of the classes. Then, using ML estimates for the class priors, plug into the Bayesian decision rule, classify the cheetah image and measure the probability of error. All of the parameters above are functions of α. Repeat the procedure for the values of α given in the file Alpha.mat. Plot the curve of the probability of error as a function of α. Can you explain the results?

B) For D<sub>1</sub>, compute the probability of error of the ML procedure identical to what we have in Method 2. Compare with the results of A). Can you explain?

C) Repeat A) with the MAP estimate of µ, i.e. using

![image](https://user-images.githubusercontent.com/15370068/147507327-6da63107-3387-46db-a647-1c18e35d3093.png)

where

![image](https://user-images.githubusercontent.com/15370068/147507341-133e87d6-b50c-44ec-9abb-0b908b964669.png)

Compare the curve with those obtained above. Can you explain the results?

D) Repeat A) to C) for each of the datasets D<sub>i</sub>, i = 2, ..., 4. Can you explain the results?

E) Repeat A) to D) under strategy 2 for the selection of the prior parameters. Comment the differences between the results obtained with the two strategies.
