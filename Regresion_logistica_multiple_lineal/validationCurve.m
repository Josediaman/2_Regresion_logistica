function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, num_labels, initial_theta,num_iters)
% lambda_vec: values of lambda.
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% num_lables: number of labels.
% num_iters: number of iterations.
% initial_theta: initial parameters.


             

lambda_vec = [0 2 4 6 8 10]';
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)   

lambda = lambda_vec(i,1);
fprintf('\n Case lambda= %f\n',lambda);
[all_theta] = oneVsAll(X, y, num_labels, lambda, num_iters,initial_theta);

a=0;
b=0;
for j = 1:num_labels
yy = (y==j);
yy2 = (yval==j);
cost1=lrCostFunction(all_theta(j,:)', X, yy, 0);
cost2=lrCostFunction(all_theta(j,:)', Xval, yy2, 0);
if(isnan(cost1)!=1) a=a+cost1; endif
if(isnan(cost2)!=1) b=b+cost2; endif
endfor

error_train(i) = a;
error_val(i) = b;

endfor


end
