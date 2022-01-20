import numpy as np
class LinearReg:
    iterations=100000
    learning_rate=0.007
    
    def init_weights(self, x):
        return np.zeros(x.shape[1]).reshape(-1,1)
    def hypothesis(self,x,w):
        y_hat=x @ w
        return y_hat
    def cost_fn(self,y,y_hat):
        m=y.shape[0]
        cost=(1/(2*m))*float(sum(np.square(y_hat-y)))
        return cost
    def gradient_descent(self,x,y,w,y_hat,alpha):
        m=y.shape[0]
        w-=(alpha/m)*(np.transpose(x) @ (y_hat-y))
        return w
    def fit_model(self,x,y,verbose):
        weights=self.init_weights(x)
        costs=[]
        for itertaion in range(self.iterations):
            weights=self.gradient_descent(x,y,weights,self.hypothesis(x,weights), self.learning_rate)
            J=self.cost_fn(y, self.hypothesis(x,weights))
            costs.append(J)
            if itertaion==int(self.iterations/4) and verbose:
                print(f"Cost after {itertaion}/{self.iterations} iterations is : {J}")
            if itertaion==int(self.iterations/2) and verbose:
                print(f"Cost after {itertaion}/{self.iterations} iterations is : {J}")
            if itertaion==int(3*self.iterations/4) and verbose:
                print(f"Cost after {itertaion}/{self.iterations} iterations is : {J}")
        print(f"Cost after {self.iterations}/{self.iterations} iterations is : {J}")
        return weights,J,costs