

%% ................................................
%% ................................................
%% LINEAL LOGISTIC REGRESSION WITH SEVERAL VARIABLES
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
%%%%%%********Select features and labels********   
input_layer_size  = 400;  
num_labels = 10;            
%%%%%%********Select archive********   
load('ex3data1.mat'); 
m = size(X, 1);
fprintf('(X,y) (10 items)\n');   
[X(1:10,1:5) y(1:10,:)]
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 3. Normalizing Features and adding first colum of ones
fprintf('Normalizing Features and adding first colum of ones ...\n');


[X mu sigma] = featureNormalize(X);
X=X(:,sigma!=0);
[m, n] = size(X); 
X = [ones(m, 1) X];
fprintf('X (normal) (10 items)\n');
X(1:10,1:5)
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





%% 4. Initial values
%%%%% *************Select initial theta and lambda***********
initial_theta = zeros(n, 1);
lambda = 1;
fprintf('Initial values: \n\n');
fprintf('Theta (first 5 values): \n');
fprintf(' %f \n', initial_theta(1:5));
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n \n \n \n');
pause;


%% 5. Run Gradient Descent with octabe function (fminunc)
fprintf('Running gradient with octabe function...\n \n');
%%%%% *************Select iterations***********
num_iters=50;
[all_theta] = oneVsAll(X, y, num_labels, lambda, num_iters,initial_theta);


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


%% 6. Display results
fprintf('\n \nTheta (first 5 rows and colums): \n');
all_theta(1:5,1:5)
fprintf('\n');
pred = predictOneVsAll(all_theta, X);
pred1 = predictOneVsAll(all_theta, Xval);
pred2 = predictOneVsAll(all_theta, Xerr);
fprintf('\nTrain Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nCross Accuracy: %f\n', mean(double(pred1 == yval)) * 100);
fprintf('\nTest Accuracy: %f\n', mean(double(pred2 == yerr)) * 100);
fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;





%% ============== Part 4: Sample to predict  ==============
fprintf('SAMPLE\n...... \n \n \n \n');





%% 7. Select a sample to predict
%%%%% *************Select sample to predict***********
x1 = X(6,:);        
sigma=sigma(sigma!=0);
mu=mu(sigma!=0);           
x2 = (x1(1,2:end).*sigma).+mu;


%% 11. Estimate the y of the sample
estimation_y = predictOneVsAll(all_theta, x1);
fprintf('Prediction:\n x= \n');
fprintf('%f  \n',x1(1, 2:8));
fprintf('...\n');
fprintf('\n y_pred= %f',estimation_y);
fprintf('\n y_real= %f \n \n',y(6));
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ==== Part 6: Learning Curve for Linear Regression ========
fprintf('\n\n LEARNING CURVE\n............... \n \n \n \n');





[error_train, error_val,ind] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  lambda, num_labels, initial_theta
,2,num_iters);


plot(ind, error_train, ind, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 7: Validation ================
fprintf('\n\nVALIDATION\n.......... \n \n \n \n');





[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, num_labels,initial_theta,num_iters);
close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
fprintf('\n\nlambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end
fprintf('\nThe best lambda has the lowest validation error.\n\n');
fprintf('\nActual lambda: %f\n\n',lambda);
fprintf('Program paused. Press enter to continue.\n');
pause;







