# Method 1
To formulate this as a pattern recognition problem, we need to decide on an observation space. Here we will be using the space of 8 × 8 image blocks, i.e. we view each image as a collection of 8 × 8 blocks. For each block we compute the discrete cosine transform (function dct2 on MATLAB) and obtain an array of 8 × 8 frequency coefficients. We do this because the cheetah and the grass have different
textures, with different frequency decompositions and the two classes should be better separated in the
frequency domain. We then convert each 8 × 8 array into a 64 dimensional vector because it is easier
to work with vectors than with arrays. The file Zig-Zag Pattern.txt contains the position (in the 1D
vector) of each coefficient in the 8 × 8 array. The file TrainingSamplesDCT 8.mat contains a training
set of vectors obtained from a similar image (stored as a matrix, each row is a training vector) for each
of the classes. There are two matrices, TrainsampleDCT BG and TrainsampleDCT FG for foreground
and background samples respectively.
To make the task of estimating the class conditional densities easier, we are going to reduce each
vector to a scalar. For this, for each vector, we compute the index (position within the vector) of the
coefficient that has the 2nd largest energy value (absolute value). This is our observation or feature X.
(The reason we do not use the coefficient with the largest energy is that it is always the so-called “DC”
coefficient, which contains the mean of the block). By building an histogram of these indexes we obtain
the class-conditionals for the two classes PX|Y (x|cheetah) and PX|Y (x|grass). The priors PY (cheetah)
and PY (grass) should also be estimated from the training set.
