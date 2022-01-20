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
  * x is the multivariable input training dataset and y is the target variable of that dataset.

* Verbose :
  * Verbose is a flag variable that can be set to  `True` to display the cost after every 1/4th of the total iteration

## Working of the model
* The input variable is taken as matrix of `m` rows and `n` columns and the target variable is taken as a vector of `m` rows.  
* Initial value of weights is taken as a vector of zeros of size `n`.
* Hypothesis is a candidate model that approximates a target function for mapping inputs to outputs.
* Initial hypothesis is calulated by performing matrix multiplication of input variable and the initial weights as shown below:

  ![eqn2](https://user-images.githubusercontent.com/84195790/150275908-86a94b9a-88f7-48fe-99c7-0074c7712faa.gif)
  
* The cost of this hypothesis is calculated using the mean squared error as shwon below:
  
  ![eqn1](https://user-images.githubusercontent.com/84195790/150277610-5444fec2-6025-4918-9652-af812e6f8673.gif)
 
* In the next step the gradient of this cost function is calculated and multiplied by the learning_rate
* The initial weight is then replaced by this gradient calculated as shown below and this process is known as gradient descent. More intuition on gradient descent will be given later in this documentation.
  
  ![eqn3](https://user-images.githubusercontent.com/84195790/150278395-6573a169-c8ad-4581-9976-2c43073712b6.gif)

* The above steps are repeated for the number of iterations that was initialized (in this case `100000`).

## Gradient Descent intuition
* Gradient descent is an iterative optimization algorithm used for finding a local minima. The idea is to take repeated steps in the opposite direction of the gradient until we reach the local minima of the cost function.
* The gradient descent algorithm is as shown below:

  ![eqn3](https://user-images.githubusercontent.com/84195790/150280507-50b25eba-425b-4926-93b0-d4d860212a41.gif)

* We know that the partial derivative of the cost function with respect to weights results in gradient equation. This gradient equation gives us the slope at a point `J(w)` on the cost vs weight graph.
* The descent of cost on the cost vs weights graph looks as shown in the image below:

  ![Graph_Plotter2](https://user-images.githubusercontent.com/84195790/150286908-0f40b29e-ce89-468b-8d66-636095b9c984.png)
  
* If the learning rate is too small, gradient descent can be slow.
* If the learning rate is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

  ![gradient-descent-divergence](https://user-images.githubusercontent.com/84195790/150288077-fb3dc67a-6f63-4446-97e8-723fab8efc77.gif)
* Gradient descent can converge to a local minima, even with the learning rate is fixed.
* As we approach a local minimum, gradient descent will automatically take smaller steps as the slope (gradient) decreases
* Each step of gradient descent uses all the training examples.
* The animation below gives the virtual representation of how the gradient descent works.

  ![68747470733a2f2f707669676965722e6769746875622e696f2f6d656469612f696d672f70617274312f6772616469656e745f64657363656e742e676966](https://user-images.githubusercontent.com/84195790/150288315-84a091fd-46d8-4211-8777-b23653ba9d17.gif)
  
## Usage

```py
from linearregression import LinearReg
import numpy as np

X2=np.random.rand(500,1)
Y2=(3*X2)+(np.random.rand(500,1)*0.1)
model=LinearReg()
weights,J,costs=model1.fit_model(X2,Y2, True)

print("Final cost: ", J)
print("Weights: ",weights)

y_pred=model.predict(X2)
```
