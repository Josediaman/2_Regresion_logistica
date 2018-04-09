function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, initial_theta, options)
% lambda_vec: values of lambda.
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% initial_theta: initial values of theta.
% options: options of the minimization.


         

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.5 1 1.5 2]';
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
         
	lambda = lambda_vec(i,1);
     [theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), 	initial_theta, options);
	error_train(i) = costFunctionReg(theta, X, y, 0);
	error_val(i) = costFunctionReg(theta, Xval, yval, 0);

end


end
