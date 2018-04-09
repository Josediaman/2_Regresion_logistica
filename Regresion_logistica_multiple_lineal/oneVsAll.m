function [all_theta] = oneVsAll(X, y, num_labels, lambda, num_iters,initial_theta)
% all_theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% num_labels: number of labels.
% lambda: Paramer of the regularization.
% num_iters: Number of iterations.
% initial_theta: initial values of theta.


m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n );
options = optimset('GradObj', 'on', 'MaxIter', num_iters);

for i = 1:num_labels
theta = zeros(n, 1);
yy = (y==i);
[theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, yy, lambda)), ...
                initial_theta, options);
all_theta(i,:)=theta';
endfor


end
