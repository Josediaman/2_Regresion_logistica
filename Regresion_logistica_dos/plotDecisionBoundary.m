function plotDecisionBoundary(theta, X, y, d,mu,sigma,n)
% theta: Parameters of the regresion.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% d: polynomial degree
% mu: Main value of X (by colums).
% sigma: Standart derivation of X (by colums).
% n: number of features.

hold on;

if size(X, 2) <= 3    
    plot_x = [min(X(:,2)),  max(X(:,2))];
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    plot(plot_x, plot_y)
  
else    
    u = linspace(min(X(:,2)), max(X(:,2)), 20);
    v = linspace(min(X(:,3)), max(X(:,3)), 20);
    z = zeros(length(u), length(v));
    
    for i = 1:length(u)
        for j = 1:length(v)
		 zzz = mapFeature(u(i), v(j), d);
		 zzz(2:n+1) = (zzz(2:n+1)-mu)./sigma;                		 z(i,j) = zzz*theta;
        end
    end
    z = z'; 
    u = (u-mu(1))/sigma(1);
    v = (v-mu(2))/sigma(2);
    contour(u, v, z, [0,0], 'LineWidth', 2);
end

hold off


end







