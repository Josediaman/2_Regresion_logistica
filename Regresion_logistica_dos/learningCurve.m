function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, initial_theta, options,poly_grade);
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% lambda: parameter of regularization.
% initial_theta: initial values of theta.
% options: options of the minimization.
% poly_grade: polynomial degree.

m = size(X, 1);
d=poly_grade+1;
k=m-d;
error_train = zeros(k, 1);
error_val   = zeros(k, 1);

for i = d:m
	x_train = X(1:i,:);
	y_train = y(1:i);
	[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, x_train, y_train, lambda)), 	initial_theta, options);
p=i-d+1;
	error_train(p) = costFunctionReg(theta, x_train, y_train, 0);
	error_val(p) = costFunctionReg(theta, Xval, yval, 0);

endfor


end
