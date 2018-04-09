function out = mapFeature(X1, X2, d)
% out: Lista te términos del polinómio de grado.
% X1: First variable 
% X2: Second variable
% d: Degree of the polynomic.

degree = d;
out = ones(size(X1(:,1)));


for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end


end