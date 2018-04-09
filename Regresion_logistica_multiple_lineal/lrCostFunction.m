function [J, grad] = lrCostFunction(theta, X, y, lambda)
% J: Cost of the regresion with theta.
% grad: Gradient of J.
% theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% lambda: Paramer of the regularization.


m = length(y); 
J = 0;
grad = zeros(size(theta));

prob=sigmoid(X*theta);
theta2=theta(2:size(theta));

J=-(1/m)*sum(y.*log(prob)+(1-y).*log(1-prob))+(lambda/(2*m))*theta2'*theta2;

grad(1)=(1/m)*sum((prob-y).*X(:,1))';
grad(2:size(theta))=(1/m)*sum((prob-y).*X(:,2:size(theta)))'+(lambda/m)*theta2;


end
