# Information:

Dataset = data.h5

training set = **m_train**
	which contains:
	    cat (y=1)
	    non-cat (y=0)

Test set = m_test
	which contains:
	labelled images as "cat" or "non-cat"

Image shape = (num_px, num_px, 3) - each image is square - 3=colorChannel(RGB)

# Classifying Pictures as Cat or Not a Cat

Goal:

## Exercise 1

Find Values for:

1. m_train
2. m_test
3. num_px

- We can access "m_train" by writing **`train_set_x_orig.shape[0]`**
- test set = test_set_x_orig.shape[0]
- num_px = (64, 64, 3)


- m_train = train_set_x_orig.shape[0]
- m_test = test_set_x_orig.shape[0]
- num_px = (64, 64, 3)

## Exercise 2

Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px **∗**∗ num_px **∗**∗ 3, 1).

A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b**∗**∗c**∗**∗d, a) is to use:

```python
X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
```

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T


To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).


Common steps for pre-processing a new dataset are:

* Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
* Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
* "Standardize" the data
