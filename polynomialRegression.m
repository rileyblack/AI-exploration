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

% extracting max and min data
xmax = max(max(xtr),max(xte));
xmin = min(min(xtr),min(xte));
ymax = max(max(ytr),max(yte));
ymin = min(min(ytr),min(yte));

% defining x range for plotting
x = linspace(xmin,xmax);

% plot training/testing data
figure()
subplot(121)
plot(xtr, ytr, 'ob');
title('Training Data');
xlabel('Features');
ylabel('Desired Outputs');
xlim([xmin xmax]);
ylim([ymin ymax]);
subplot(122)
plot(xte, yte, 'ob');
title('Testing Data');
xlabel('Features');
ylabel('Desired Outputs');
xlim([xmin xmax]);
ylim([ymin ymax]);

% creating containers to track training/testing error for each regression
trainingerr = zeros(4,1);
testingerr = zeros(4,1);


%--------------------------------------------------------------------------
%1ST ORDER REGRESSION (LINEAR REGRESSION)
%--------------------------------------------------------------------------

% appending column of 1's to training data
X1 = [xtr ones(numtr,1)];

% calculating 1st order regression 2D weight vector 
w1 = (X1'*X1)\(X1'*ytr);

% extracting 1st order regression slope & intercept
m1 = w1(1);
b1 = w1(2);

% calculating predicted labels for training/testing data
wTx1 = m1*xtr + b1;
wTx2 = m1*xte + b1;

% calculating average error for training/testing
trainingerr(1) = immse(wTx1,ytr);
testingerr(1) = immse(wTx2,yte);

% creating 1st order regression line for plotting
y1 = m1*x + b1;

% plotting training/testing data and 1st order regression line
figure()
subplot(121)
plot(xtr, ytr, 'ob');
title('Training Data and 1st Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y1,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;
subplot(122)
plot(xte, yte, 'ob');
title('Testing Data and 1sr Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y1,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;


%--------------------------------------------------------------------------
%2ND ORDER REGRESSION
%--------------------------------------------------------------------------

% adding column of x^2 features to training data
X2 = [X1(1:end,1).^2 X1];

% calculating 2nd order regression 3D weight vector
w2 = (X2'*X2)\(X2'*ytr);

% extracting 2nd order regression polynomial constants
a2 = w2(1);
b2 = w2(2);
c2 = w2(3);

% calculating predicted labels for training/testing data
wTx3 = (a2*(xtr.^2))+(b2.*xtr)+c2;
wTx4 = (a2*(xte.^2))+(b2.*xte)+c2;

% calculating average error for training/testing data
trainingerr(2) = immse(wTx3,ytr);
testingerr(2) = immse(wTx4,yte);

% creating 2nd order regression line for plotting
y3 = (a2*(x.^2))+(b2.*x)+c2;

% plotting training/testing data and 2nd order regression line
figure()
subplot(121)
plot(xtr, ytr, 'ob');
title('Training Data and 2nd Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y3,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;
subplot(122)
plot(xte, yte, 'ob');
title('Testing Data and 2nd Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y3,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;


%--------------------------------------------------------------------------
%3RD ORDER REGRESSION
%--------------------------------------------------------------------------

% adding column of x^3 features to training data
X3 = [X1(1:end,1).^3 X2];

% calculating 3rd order regression 4D weight vector
w3 = (X3'*X3)\(X3'*ytr);

% extracting 3rd order regression polynomial constants
a3 = w3(1);
b3 = w3(2);
c3 = w3(3);
d3 = w3(4);

% calculating predicted labels for training/testing data
wTx5 = (a3*(xtr.^3))+(b3*(xtr.^2))+(c3.*xtr)+d3;
wTx6 = (a3*(xte.^3))+(b3*(xte.^2))+(c3.*xte)+d3;

% calculating average error for training/testing data
trainingerr(3) = immse(wTx5,ytr);
testingerr(3) = immse(wTx6,yte);

% creating 3rd order regression line for plotting
y5 = (a3*(x.^3))+(b3*(x.^2))+(c3.*x)+d3;

% plotting training/testing data and 3rd order regression line
figure()
subplot(121)
plot(xtr, ytr, 'ob');
title('Training Data and 3rd Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y5,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;
subplot(122)
plot(xte, yte, 'ob');
title('Testing Data and 3rd Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y5,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;


%--------------------------------------------------------------------------
%4TH ORDER REGRESSION
%--------------------------------------------------------------------------

% adding column of x^4 features to training data
X4 = [X1(1:end,1).^4 X3];

% calculating 4th order regression 5D weight vector 
w4 = (X4'*X4)\(X4'*ytr);

% extracting 4th order regression polynomial constants
a4 = w4(1);
b4 = w4(2);
c4 = w4(3);
d4 = w4(4);
e4 = w4(5);

% calculating predicted labels for training/testing data
wTx7 = (a4*(xtr.^4))+(b4*(xtr.^3))+(c4*(xtr.^2))+(d4.*xtr)+e4;
wTx8 = (a4*(xte.^4))+(b4*(xte.^3))+(c4*(xte.^2))+(d4.*xte)+e4;

% calculating average error for training/testing data
trainingerr(4) = immse(wTx7,ytr);
testingerr(4) = immse(wTx8,yte);

% creating 4th order regression line for plotting
y7 = (a4*(x.^4))+(b4*(x.^3))+(c4*(x.^2))+(d4.*x)+e4;

% plotting training data and 4th order regression line
figure()
subplot(121)
plot(xtr, ytr, 'ob');
title('Training Data and 4th Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y7,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;
subplot(122)
plot(xte, yte, 'ob');
title('Testing Data and 4th Order Regression Line');
xlabel('Features');
ylabel('Desired Outputs');
hold on;
plot(x,y7,'-r');
xlim([xmin xmax]);
ylim([ymin ymax]);
hold off;


%--------------------------------------------------------------------------
%COMPARISON PLOTS
%--------------------------------------------------------------------------

% plotting training/testing errors to compare
figure();
subplot(121)
plot(1:1:4, trainingerr, '-o');
title('Average Training Error');
xlabel('Regression Degree');
ylabel('Average Error');
xlim([0 5]);
xticks(1:1:4);
ylim([0 3]);
subplot(122)
plot(1:1:4, testingerr, '-o');
title('Average Testing Error');
xlabel('Regression Degree');
ylabel('Average Error');
xlim([0 5]);
xticks(1:1:4);
ylim([0 3]);
