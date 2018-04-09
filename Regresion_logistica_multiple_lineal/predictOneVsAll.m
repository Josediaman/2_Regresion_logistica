function p = predictOneVsAll(all_theta, X)
% p: Predictions.
% all_theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.


K = size(all_theta, 1);
m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
[maxi ix]=max(sigmoid(X*all_theta'),[],2);
p=ix;


end
