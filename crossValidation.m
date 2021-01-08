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

% creating matrix to not penalize bias term
I = eye(5);
I(5,5) = 0;


%--------------------------------------------------------------------------
%5-FOLD CROSS VALIDATION
%--------------------------------------------------------------------------

% computing size of each "fold" in 5 fold validation
foldsize = numtr/5;

% creating containers to track training/testing error for each parameter
trainingerrcrossval = zeros(7,1);
testingerrcrossval = zeros(7,1);

% iterating through each regularization parameter
for i = 1:7
    
    % defining container to track training/testing error for each "fold"
    foldtrainingerr = zeros(5,1);
    foldtestingerr = zeros(5,1);
    
    % iterating through each "fold"
    for j = 0:4
        
        % computing current validation set location in training set 
        foldstartindex = 1 + (j*foldsize);
        foldendindex = (foldstartindex + (foldsize-1));
        
        % extracting validation set
        vx = xtr(foldstartindex:foldendindex);
        vy = ytr(foldstartindex:foldendindex);
        
        % extracting training set
        tx = xtr;                                   % begin with full x set
        ty = ytr;                                   % begin with full y set
        tx(foldstartindex:foldendindex) = [];       % remove validation
        ty(foldstartindex:foldendindex) = [];       % remove validation
        
        % compute 4th order inputs
        Xt = [tx.^4 tx.^3 tx.^2 tx ones(length(tx),1)];
        
        % compute 5D weight vector
        wt = ((Xt'*Xt)+(lambda(i)*I))\(Xt'*ty);

        % extracting 4th order regression polynomial constants
        a = wt(1);
        b = wt(2);
        c = wt(3);
        d = wt(4);
        e = wt(5);

        % calculating predicted labels for training/testing data
        trainingTx = (a*(tx.^4))+(b*(tx.^3))+(c*(tx.^2))+(d.*tx)+e;
        testingTx = (a*(vx.^4))+(b*(vx.^3))+(c*(vx.^2))+(d.*vx)+e;

        % calculating average error for training/testing data over fold
        foldtrainingerr(j+1) = immse(trainingTx,ty);
        foldtestingerr(j+1) = immse(testingTx,vy);
    end
    
    % calculating average error for training/testing data over parameter
    trainingerrcrossval(i) = mean(foldtrainingerr);
    testingerrcrossval(i) = mean(foldtestingerr);
end


%--------------------------------------------------------------------------
%TRADITIONAL TRAINING
%--------------------------------------------------------------------------

% creating containers to track training/testing error for each parameter
trainingerrtraditional = zeros(7,1);
testingerrtraditional = zeros(7,1);

% iterating through each regularization parameter
for i = 1:7
    
    % compute 4th order inputs
    X = [xtr.^4 xtr.^3 xtr.^2 xtr ones(numtr,1)];
    
    % compute 5D weight vector 
    w = ((X'*X)+(lambda(i)*I))\(X'*ytr);
    
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
    trainingerrtraditional(i) = immse(trainingwTx,ytr);
    testingerrtraditional(i) = immse(testingwTx,yte);  
end


%--------------------------------------------------------------------------
%PLOTTING COMPARISON
%--------------------------------------------------------------------------

figure()
subplot(121)
semilogx(lambda, trainingerrcrossval, '-o')
title('Five-Fold Cross-Validation')
xlabel('Lambda')
ylabel('Average Error')
hold on;
semilogx(lambda, testingerrcrossval, '-o');
legend('Training Data Error','Testing Data Error');
hold off;
subplot(122)
semilogx(lambda, trainingerrtraditional, '-o')
title('Traditional')
xlabel('Lambda')
ylabel('Average Error')
hold on;
semilogx(lambda, testingerrtraditional, '-o');
legend('Training Data Error','Testing Data Error');
hold off;
