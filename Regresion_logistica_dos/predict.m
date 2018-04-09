function p = predict(theta, X)
% p: Value of training examples according to model.
% theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.


m = size(X, 1); 
p = zeros(m, 1);
p=(X*theta>=0);


end
