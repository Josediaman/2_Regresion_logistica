function [error_train, error_val,ind] = ...
    learningCurve(X, y, Xval, yval, lambda, num_labels,initial_theta,poly_grade,num_iters)
% error_train: error of train set.
% error_val: error of cross validation set.
% indef of number of examples.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% lambda: parameter of regularization.
% num_labels: number of labels.
% initial_theta: initial values of theta.
% poly_grade: polynomial degree.
% num_iters: number of iterations.

m = size(X, 1);
num=5;
error_train = zeros(num+1, 1);
error_val   = zeros(num+1, 1);
valu=floor(m/num)-2;
ind=0;

for i=1:num+1,
	
	value=(i-1)*valu+2;
	ind=[ind value];
	fprintf('\n Case m= %f\n',ind(i+1));
	x_train = X(1:value,:);
	y_train = y(1:value);
	[all_theta] = oneVsAll(x_train, y_train, num_labels,         lambda, num_iters,initial_theta);
	a=0;
	b=0;
	for j = 1:num_labels
	yy = (y_train(1:value)==j);
	yy2 = (yval==j);
	cost1=lrCostFunction(all_theta(j,:)', x_train, yy, 0);
	cost2=lrCostFunction(all_theta(j,:)', Xval, yy2, 0);
	if(isnan(cost1)!=1) a=a+cost1; endif
	if(isnan(cost2)!=1) b=b+cost2; endif
	endfor
	error_train(i) = a;
	error_val(i) = b;

endfor

ind=ind(2:num+2);

error_train
error_val



end
