# Method 4
In this method we use the cheetah image to evaluate the performance of a classifier based on mixture models estimated with EM. Once again we use the decomposition into 8×8 image blocks, compute the DCT of each block, and zig-zag scan. For this (using the data in TrainingSamplesDCT_new_8.mat) we fit a mixture of Gaussians of diagonal covariance to each class, i.e.
![image](https://user-images.githubusercontent.com/15370068/147961372-39f46e8d-f5d5-48b6-b142-9dcc391e92bd.png)

where all Σ<sub>c</sub> are diagonal matrices. We then apply the BDR based on these density estimates to the cheetah image and measure the probability of error as a function of the number of dimensions of the space (as before, use {1, 2, 4, 8, 16, 24, 32, . . . , 64} dimensions).

A) For each class, learn 5 mixtures of C = 8 components, using a random initialization (recall that the mixture weights must add up to one). Plot the probability of error vs. dimension for each of the 25 classifiers obtained with all possible mixture pairs. Comment the dependence of the probability of error on the initialization.

B) For each class, learn mixtures with C ∈ {1, 2, 4, 8, 16, 32}. Plot the probability of error vs. dimension for each number of mixture components. What is the effect of the number of mixture components on the probability of error?

### Results:
![image](https://user-images.githubusercontent.com/15370068/147961920-7a8dd3e8-352e-4673-aa77-b2f93258260a.png)

![image](https://user-images.githubusercontent.com/15370068/147961741-75e4f9b1-29d8-4c64-837b-2aee783ef33c.png)

![image](https://user-images.githubusercontent.com/15370068/147961782-d1b594d4-b9fb-4e57-a1eb-3ecb26e6a2d7.png)
