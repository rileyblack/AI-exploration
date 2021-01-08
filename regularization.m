% developed by Riley Black

% initializing MATLAB environment
close all;
clc;
clear all;

% loading data
ytr = load('hw1ytr.dat');           % training labels
xtr = load('hw1xtr.dat');           % training data
yte = load('hw1yte.dat');           % testing labels
xte = load('hw1xte.dat');           % testing data

% counting data
numtr = length(xtr);
numte = length(xte);

% creating set of regularization parameters to test
lambda = [0.01, 0.1, 1, 10, 100, 1000, 10000];

% creating inputs for 4th order regression
X = [xtr.^4 xtr.^3 xtr.^2 xtr ones(numtr,1)];

% creating matrix to not penalize bias term
I = eye(5);
I(5,5) = 0;

% creating containers to track training/testing error for each parameter
trainingerr = zeros(7,1);
testingerr = zeros(7,1);

% creating container to hold weights for each regularization parameter
ws = zeros(5,7);

% iterating through each regularization parameter
for i = 1:7
    
    % calculating 4th order regression 5D weight vector 
    w = ((X'*X)+(lambda(i)*I))\(X'*ytr);
    
    % recording weights
    ws(:,i) = w;
    
    % extracting 4th order regression polynomial constants
    a = w(1);
    b = w(2);
    c = w(3);
    d = w(4);
    e = w(5);
    
    % calculating predicted labels for training/testing data
    trainingwTx = (a*(xtr.^4))+(b*(xtr.^3))+(c*(xtr.^2))+(d.*xtr)+e;
    testingwTx = (a*(xte.^4))+(b*(xte.^3))+(c*(xte.^2))+(d.*xte)+e;
    
    % calculating average error for training/testing data
    trainingerr(i) = immse(trainingwTx,ytr);
    testingerr(i) = immse(testingwTx,yte);  
end

% plotting results
figure();
subplot(121)
semilogx(lambda, trainingerr, '-o');
title('Average Error vs. Regularization Parameter');
xlabel('Regularization Parameter');
ylabel('Average Error');
hold on;
semilogx(lambda, testingerr, '-o');
legend('Training Data Error','Testing Data Error');
hold off;
subplot(122)
semilogx(lambda, ws(:, 1:7), '-o');
title('Weight Parameters vs. Regularization Parameter');
xlabel('Regularization Parameter');
ylabel('Weight Parameter Value');
legend('w4', 'w3', 'w2', 'w1', 'w0');
