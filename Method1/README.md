# Method 1
To formulate this as a pattern recognition problem, we need to decide on an observation space. Here we will be using the space of 8 × 8 image blocks, i.e. we view each image as a collection of 8 × 8 blocks. For each block we compute the discrete cosine transform (function dct2 on MATLAB) and obtain an array of 8 × 8 frequency coefficients. We do this because the cheetah and the grass have different textures, with different frequency decompositions and the two classes should be better separated in the frequency domain. We then convert each 8 × 8 array into a 64-dimensional vector because it is easier to work with vectors than with arrays. The file Zig-Zag Pattern.txt contains the position (in the 1D vector) of each coefficient in the 8 × 8 array. The file TrainingSamplesDCT_8.mat contains a training set of vectors obtained from a similar image (stored as a matrix, each row is a training vector) for each of the classes. There are two matrices, TrainsampleDCT_BG and TrainsampleDCT_FG for foreground and background samples respectively. 

To make the task of estimating the class conditional densities easier, we are going to reduce each vector to a scalar. For this, for each vector, we compute the index (position within the vector) of the coefficient that has the 2nd largest energy value (absolute value). This is our observation or feature X. (The reason we do not use the coefficient with the largest energy is that it is always the so-called “DC” coefficient, which contains the mean of the block). By building a histogram of these indices we obtain the class-conditionals for the two classes P<sub>X|Y</sub>(x|cheetah) and P<sub>X|Y</sub>(x|grass). The priors P<sub>Y</sub>(cheetah) and P<sub>Y</sub>(grass) should also be estimated from the training set.

**A) Using the training data in TrainingSamplesDCT_8.mat, what are reasonable estimates for the prior probabilities?**

A reasonable estimate for the prior probabilities can be obtained by using the number of training samples for each class.

**B) Using the training data in TrainingSamplesDCT_8.mat, compute and plot the index histograms P<sub>X|Y</sub>(x|cheetah) and P<sub>X|Y</sub>(x|grass).**

The histogram for the two classes is shown below. Notice that there is a significant amount of overlap, indicating that the feature we are using (index of the 2nd larget DCT coefficient) is not very good.

![hist_screenshot](https://user-images.githubusercontent.com/15370068/147173825-a914fd6e-5b70-4912-bd5e-b8fc7af85ed2.png)

**C) For each block in the image cheetah.bmp, compute the feature X (index of the DCT coefficient with 2nd greatest energy). Compute the state variable Y using the minimum probability of error rule based on the probabilities obtained in A) and B). Store the state in an array A. Using the commands imagesc and colormap(gray(255)) create a picture of that array.**

The minimum probability of error segmentation mask is shown below. Notice that the segmentation is quite noisy, confirming what one would expect from the histograms above.

![method1](https://user-images.githubusercontent.com/15370068/147173986-38e490b9-78d8-4e64-a7de-d9396c55dbaa.png)

**D) The array A contains a mask that indicates which blocks contain grass and which contain the cheetah. Compare it with the ground truth provided in image cheetah_mask.bmp and compute the probability of error of your algorithm.**

The probability of error is 17.25%

