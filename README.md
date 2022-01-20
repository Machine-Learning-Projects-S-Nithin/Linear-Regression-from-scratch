# Linear-Regression-from-scratch
A simple Linear Regression class build from scratch using numpy and visualized using matplotlib

## Initializations
* ### learning_rate:
  * determines the step size at each iteration while moving toward a minimum of the cost function. A high value for this can cause overshooting, causing the model to be inaccurate and a low value could result in smaller steps towards local-minima, meaning slow convergence and requiring more iterations.
  * Default value: `0.007`
* ### iterations:
  * the number of steps towards the local-minima. A high value for this simply increases the complexity of this model and a low value doesn't take us close to the local-minima.
  * Default value: `100000`

## Parameters
Parameters of this model include
* x and y :
  * x is the multivariable input training dataset and y is the target variable for that dataset
* Verbose :
  * Verbose is a flag variable that can be set to  `True` to display the cost after every 1/4th of the total iteration

## Working of the model
* The input variable is taken as matrix of `m` rows and `n` columns and the target variable is taken as a vector of `m` rows.  
* Initial value of weights is taken as a vector of zeros of size `n`.
* Hypothesis is a candidate model that approximates a target function for mapping inputs to outputs.
* Initial hypothesis is calulated by performing matrix multiplication of input variable and the initial weights as shown below.

  ![eqn2](https://user-images.githubusercontent.com/84195790/150275908-86a94b9a-88f7-48fe-99c7-0074c7712faa.gif)
  
* The cost of this hypothesis is calculated using the mean squared error as shwon below:
  
  ![eqn1](https://user-images.githubusercontent.com/84195790/150277610-5444fec2-6025-4918-9652-af812e6f8673.gif)
 
* In the next step the gradient of this cost function is calculated and multiplied by the learning_rate
* The initial weight is then replaced by this gradient calculated as shown below and this process is known as gradient descent. More intuition on gradient descent will be given later in this documentation.
  
