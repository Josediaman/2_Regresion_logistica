function [XX, yy, Xval, yval, Xerr, yerr, m, n] = ...
    selectsets(X, y)
% XX: X train set.
% yy: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% Xerr: X test set.
% yerr: y test set.
% m: number of examples.
% n: number of characteristics.
% X: Training examples of the data whithout feature y.
% y: Feature to predict.


num_train=floor(size(X,1)*0.6);
num_cross=floor(num_train/3);
sel = randperm(size(X, 1));
seltrain = sel(1:num_train);
selcross = sel(num_train+1:num_train+num_cross);
selerror = sel(num_train+num_cross+1:end);

% Train Set
XX=X(seltrain,:);
yy=y(seltrain,:);
[m , n] = size(XX);

% Cross Validation Set
Xval=X(selcross,:);
yval=y(selcross,:);

% Test Set
Xerr=X(selerror,:);
yerr=y(selerror,:);

end
