# Method 2
Once again we use the decomposition into 8 × 8 image blocks, compute the DCT of each block, and zig-zag scan. However, we are going to assume that the class-conditional densities are multivariate Gaussians of 64 dimensions.

Note: The training examples we used in Method 1 contained the absolute value of the DCT coefficients instead of the coefficients themselves. Please download the file TrainingSamplesDCT_8_new.mat and use it in this and all subsequent methods. For simplicity, I will still refer to it as TrainingSamplesDCT_8.mat.

**A) Using the training data in TrainingSamplesDCT_8.mat compute the maximum likelihood estimate for the prior probabilities: P<sub>Y</sub>(i), i ∈ {cheetah, grass}.**

The ML estimates of the prior probabilities are just the relative frequencies of the two events. This is exactly the same as what we did in Method 1.

**B) Using the training data in TrainingSamplesDCT_8.mat, compute the maximum likelihood estimates for the parameters of the class conditional densities P<sub>X|Y</sub>(x|cheetah) and P<sub>X|Y</sub>(x|grass) under the Gaussian assumption. Denoting by X = {X1, . . . , X64} the vector of DCT coefficients, create 64 plots with the marginal densities for the two classes - P<sub>Xk|Y</sub>(xk|cheetah) and P<sub>Xk|Y</sub>(xk|grass), k = 1, . . . , 64 - on each. Use different line styles for each marginal. Select, by visual inspection, what you think are the best 8 features for classification purposes and what you think are the worst 8 features.**

The idea here is to look for the features that have marginals, under the two classes, that are as distinct as possible. This is easy to see for the DC coefficient (X1), since the average value of image blocks of different materials is usually not the same (and the marginals are therefore somewhat separated). It is a little harder to see for coefficients that have zero mean i.e. all the others. For these, the amount of overlap is always significant and good features are ones for which one of the Gaussians is as concentrated around zero as possible and the other is as wide as possible. Below is the plot of the marginals for all 64 features. It appears that the best (most discriminant) features would be {1, 11, 14, 23, 25, 27, 32, 40} while the worst (least discriminant) are {3, 4, 5, 59, 60, 62, 63, 64}.

**C) Compute the Bayesian decision rule and classify the locations of the cheetah image using i) the 64-dimensional Gaussians, and ii) the 8-dimensional Gaussians associated with the best 8 features. For the two cases, plot the classification masks and compute the probability of error by comparing with cheetah_mask.bmp. Can you explain the results?**

Using all 64 features, the probability of error is 8.98%.
Using the 8 best features, the probability of error is 5.43%.

As the dimension of the space increases, there are two forces at play. On one hand, we have more features and more information about the image, so the classification improves. On the other hand, in high-dimensions one needs more points to have good density estimates. Hence, given a fixed number of training points, the models for the class conditional densities become worse approximations to the underlying distributions as the dimension increases. These results indicate that, for the cheetah image, the optimal point is closer to 8 than 64 dimensions.



Marginal densities:
![method2_features](https://user-images.githubusercontent.com/15370068/147176535-72350e4f-4d80-448f-ac7f-2c67e0fb830c.png)

64 features:

![method2_64](https://user-images.githubusercontent.com/15370068/147176504-35f8a893-ac24-4429-a6a8-6f920537f1dc.png)

8 features:

![method2_8](https://user-images.githubusercontent.com/15370068/147176442-2905f4c7-d6bc-4db1-bc80-d3e42a90a675.png)
