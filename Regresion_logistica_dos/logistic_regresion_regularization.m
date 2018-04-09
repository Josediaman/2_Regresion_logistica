
%% ................................................
%% ................................................
%% POLINOMIC LOGISTIC REGRESSION WITH TWO VARIABLES
%% ................................................
%% ................................................





%% 1. Clear and Close Figures
clear ; close all; clc





%% ==================== Part 1: Data ====================
fprintf('\n \nDATA\n.... \n \n \n');   





%% 2. Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add your own file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Loading data ...\n');   
%%%%%%********Select archive********   
data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
X = data(:, [1, 2]); 
y = data(:, 3);
fprintf('(X,y) (10 items)\n');   
[X(1:10,:) y(1:10,:)]
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 3. Plotting Data
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ... 
'indicating (y = 0) examples.\n']);
plotData(X, y);
hold on;
xlabel('x1')
ylabel('x2')
legend('y = 1', 'y = 0')
hold off;
fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;


%% 4. No lineal regression 
%%%%% ************* Select grade of polynomial ***********
poly_grade=6;
X = mapFeature(X(:,1), X(:,2),poly_grade);
XX=X;


%% 5. Normalizing Features and adding first colum of ones
fprintf('Normalizing Features and adding first colum of ones ...\n');
[m, n] = size(X); 
[X(:,2:n) mu sigma] = featureNormalize(X(:,2:n));
fprintf('X (normal) (10 items)\n');
n=n-1;
X(1:10,1:4)
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 4. Select train, cross and test validation sets
[X, y, Xval, yval, Xerr, yerr, m, n] = ...
    selectsets(X, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% ====== Part 2: Regularized Logistic Regression =========
fprintf('REGULARIZED LOGISTIC REGRESSION\n............................... \n \n \n \n');





%% 6. Initial values
%%%%% *************Select initial theta and lambda***********
initial_theta = zeros(n, 1);
lambda = 1;
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Initial values: \n\n');
fprintf('Theta (first 5 values): \n');
fprintf(' %f \n', initial_theta(1:5));
fprintf('\n');
fprintf('Cost: \n');
fprintf(' %f \n', cost);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 7. Run Gradient Descent with octabe function (fminunc)
fprintf('Running gradient with octabe function...\n \n');
%%%%% *************Select iterations***********
num_iters = 1000;
options = optimset('GradObj', 'on', 'MaxIter', num_iters);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract theta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional: Execute your own gradient descent.

%fprintf('Running gradient descent with alpha ... \n \n ');
%%%%% *************Select iterations***********
%num_iters = 1000;
%[theta, J_his] = gradientDescentMulti(X, y, theta, alpha, %num_iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[cost_train, grad0] = costFunctionReg(theta, X, y, 0);
[cost_cross, grad0] = costFunctionReg(theta, Xval, yval, 0);
[cost_error, grad0] = costFunctionReg(theta, Xerr, yerr, 0);


%% 8. Display results
fprintf('Theta (first 5 values): \n');
fprintf(' %f \n', theta(1:5));
fprintf('\n');
fprintf('Cost: \n');
fprintf(' %f \n', J);
fprintf('\n');
fprintf('Cost train: \n');
fprintf(' %f \n', cost_train);
fprintf('\n');
fprintf('Cost cross: \n');
fprintf(' %f \n', cost_cross);
fprintf('\n');
fprintf('Cost test: \n');
fprintf(' %f \n', cost_error);
fprintf('\n');
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\n');
fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;





%% ================ Part 2': GRAPHIC ================
fprintf('GRAPHIC \n...... \n \n \n \n');





%% 9. Plotting Data
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ... 
'indicating (y = 0) examples with the boundary.\n']);
plotData(X(:,2:3), y);
plotDecisionBoundary(theta, XX, y, poly_grade,mu,sigma,n-1);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('x1')
ylabel('x2')
legend('y = 1', 'y = 0', 'Decision boundary')
axis([min(X(:,2)), max(X(:,2)), min(X(:,3)), max(X(:,3))])
hold off;
fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;





%% ============== Part 3: Sample to predict  ==============
fprintf('SAMPLE\n...... \n \n \n \n');





%% 10. Select a sample to predict
%%%%% *************Select sample to predict***********
x1 = X(6,:);                   
x2 = (x1(1,2:end).*sigma).+mu;

 
%% 11. Estimate the y of the sample
estimation_y = sigmoid(x1 * theta);
fprintf('Probability of (y=1) of the sample:\n x_pred= ');
fprintf('%f  ',x2(1,:));
fprintf('\n y_pred= %f \n \n',estimation_y);
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ==== Part 5: Learning Curve for Linear Regression ========
fprintf('\n\n LEARNING CURVE\n............... \n \n \n \n');





[error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  lambda, initial_theta, options,poly_grade);

d=poly_grade+1;
figure;
plot(d:m, error_train, d:m, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 5: Validation ================
fprintf('\n\nVALIDATION\n.......... \n \n \n \n');





[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, initial_theta, options);

figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Error Train', 'Error Cross');
xlabel('lambda');
ylabel('Error');
fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end


fprintf('\n Actual lambda: \n');
fprintf(' %f \n', lambda);
fprintf('\nThe best lambda has the lowest validation error.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;






