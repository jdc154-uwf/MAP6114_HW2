# code used the ML logistic regression lab as a base 
# inserting my own logistic regression class to replace 
# scikit's. 
# To use scikit method, just uncomment sections specifying scikit 
# method and comment out my method.
import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#make my own class to replace the scikit logistic regression
class myLogisticRegression:

    def __init__(self):
        self.learning_rate = 1e-1 #hard coding learning rate
        self.n_iters = 5000 #hard coding number iterations
        self.w = None
        self.b = None

    def sigma(self, z):
        # pg143 scikit book logistic sigmoid function: sigma(z) = 1 / (1 + exp(-z))
        z = np.clip(z, -500, 500) #stop z from possibly overflowing
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape

        # Initialize weights
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):

            # pg143 scikit book p=sigma(xT*theta) where sigma is the logistic function xT*theta is my hypothesis (ie. xw+b) 
            # as scikit uses theta instead of w and theta_0 (first element of theta) for b. I like w and b for readability for me.
            z = X @ self.w + self.b
            p = self.sigma(z)

            # Gradients
            dw = (1/m) * (X.T @ (p - y))
            db = (1/m) * np.sum(p - y)

            # Update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_probability(self, X):
        z = X @ self.w + self.b
        return self.sigma(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_probability(X)
        return (probs >= threshold).astype(int)


# x data contains [income, savings]
# y data contains yes/no binary classifier
X_train = np.array([[30000, 90000], [67000, 49000], [37000, 24000], [91000, 74000], [69000, 93000]])
y_train = np.array([0, 1, 0, 1, 1])
x_test=np.array([[33000, 86000],[82000, 104000]])
y_test=np.array([0, 1])
#Prior to adding in this test set (and thus reducing my training set by 1-2), figure out why my solution is not matching scikit learn model even closely
#add in normalization of the data
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_train_z = (X_train - X_mean) / X_std

print("X_train:\n", X_train)
print("y_train:\n", y_train)    
print("X_test:\n", x_test)
print("y_test:\n", y_test)  
X_train[:, 0]

# Training
#lr_model = LogisticRegression() #scikit method
lr_model = myLogisticRegression()
lr_model.fit(X_train_z, y_train)

# plot training data
df = pd.DataFrame({'x1': X_train[:,0], 'x2': X_train[:,1], 'y': y_train})
ax = sns.scatterplot(data=df, x='x1', y='x2', hue='y')
ax.set_title("Training Data")
plt.show()

# Retrieve the model parameters.
 
#b = lr_model.intercept_[0] #scikit method
b = lr_model.b

#w1, w2 = lr_model.coef_.T #scikit method
w1, w2 = lr_model.w
print("Model parameters in z-space:")
print("w1 (income weight):", w1)    
print("w2 (savings weight):", w2)
print("b (intercept):", b)
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
#xmin, xmax = 25000, 105000
# plot boundary in z-space
xmin, xmax = X_train_z[:,0].min()-0.5, X_train_z[:,0].max()+0.5
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k--')
plt.scatter(X_train_z[:,0], X_train_z[:,1], c=y_train)
plt.xlabel("income (z)")
plt.ylabel("savings (z)")
plt.show()

#Test Data Predictions
x_test_z = (x_test - X_mean) / X_std
predictions = lr_model.predict(x_test_z)
print("Predictions on test set:", predictions)

# New applicant Decision
# if this was more than just 1 applicant I'd make a 
# new function like applicant_decision that returns 
# predicted decision
new_applicant = np.array([[5000, 5000]])
#prediction = lr_model.predict(new_applicant)
new_applicant_z = (new_applicant - X_mean) / X_std
prediction = lr_model.predict(new_applicant_z)[0]
#print("Predicted Decision:", prediction[0])
prediction = lr_model.predict(new_applicant)[0]

if prediction == 1:
    print("Approval Decision: YES")
else:
    print("Approval Decision: NO")

#Looks like I just needed to plot in z-space instead of copying the plot from the original base lab in class. 
# Atleast now my boundary looks correct being between my data points marking the boundary between no and yes.
#other differences with scikit learn may just be not having penalty=None as their default uses L2 rgularization