function [X_norm, mu, sigma] = featureNormalize(X)
% X_norm: X normalization (by colums).
% mu: Main value of X (by colums).
% sigma: Standart derivation of X (by colums).
% X: Training examples of the data whithout feature y.

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
