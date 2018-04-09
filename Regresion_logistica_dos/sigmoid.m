function g = sigmoid(z)
% g: value of sigmoid function in z.
% z: value/set of values. 


g = 1.0 ./ (1.0 + exp(-z));

end
