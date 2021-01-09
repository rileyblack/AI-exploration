% developed by Riley Black

% initializing MATLAB environment
close all;
clc;
clear all;

% loading image dataset into workspace
dataset = importdata('faces.dat');

% computing mean-centred image dataset
dataset_mean = mean2(dataset);
centred_dataset = dataset - dataset_mean;


%--------------------------------------------------------------------------
% visualizing dataset
%--------------------------------------------------------------------------

% extracting 100th image from each dataset
image_100 = dataset(100,:);
centred_image_100 = centred_dataset(100,:);

% reshaping each 100th image into displayable format
image_100_matrix = reshape(image_100, 64, 64);
centred_image_100_matrix = reshape(centred_image_100, 64, 64);

% displaying each 100th image
figure()
subplot(121)
imshow(image_100_matrix, [0 255])
title('Original 100th Image');
subplot(122)
imshow(centred_image_100_matrix, [0 255])
title('Mean-Centred 100th Image');


%--------------------------------------------------------------------------
% preforming pricipal component analysis
%--------------------------------------------------------------------------

% preforming PCA
[coeff, ~, latent] = pca(centred_dataset);


%--------------------------------------------------------------------------
% analyzing varience captured per # of principal components retained
%--------------------------------------------------------------------------

% computing total varience 
total_varience = sum(latent);

% defining variable to track varience captured by i first eigenvalues
running_varience = 0;

% defining container to store tracking results 
running_varience_history = zeros(length(latent), 1);

% defining variable to hold eigenvalue number capturing 95% of varience
cutoff_eigenvalue = 0;

% defining variable to flag if eigenvalue number has already been found
found_cutoff = 0;

% iterating through each eigenvalue until 95% of varience is captured 
for i = 1:length(latent)
    running_varience = running_varience + latent(i, 1);                     % updating varience currently captured
    running_varience_history(i+1, 1) = running_varience/total_varience;     % tracking varience currently captured
    if (running_varience_history(i+1, 1) > 0.95) && (found_cutoff == 0)     % checking if 95% of total varience is currently captured
        cutoff_eigenvalue = i;                                              % saving eigenvalue number capturing 95% of varience
        found_cutoff = 1;                                                   % updating found flag
    end
end

% plotting eigenvalues of the covariance matrix of the centred dataset
figure()
subplot(121)
plot(latent,'LineWidth', 2)
grid on;
title('Eigenvalues');
xlabel('Order');
ylabel('Magnitude');

% plotting percentage of variance captured per eigenvalue
subplot(122)
plot(running_varience_history,'LineWidth',2)
hold on
yline(0.95)
hold on
xline(cutoff_eigenvalue)
hold on
plot(cutoff_eigenvalue, 0.95, 'ro')
grid on;
title('Percentage of Varience Captured by First "d" Eigenvalues');
xlabel('d');
ylabel('Percentage of Varience Captured');


%--------------------------------------------------------------------------
% visualizing principal components with 5 largest variences
%--------------------------------------------------------------------------

% reshaping top 5 pricipal component into displayable format
pc1 = reshape(coeff(:, 1), 64, 64);
pc2 = reshape(coeff(:, 2), 64, 64);
pc3 = reshape(coeff(:, 3), 64, 64);
pc4 = reshape(coeff(:, 4), 64, 64);
pc5 = reshape(coeff(:, 5), 64, 64);

% displaying top 5 pricipal component
figure()
subplot(321)
imshow(pc1, [])
title('First Principal Component');
subplot(322)
imshow(pc2, [])
title('Second Principal Component');
subplot(323)
imshow(pc3, [])
title('Third Principal Component');
subplot(324)
imshow(pc4, [])
title('Fourth Principal Component');
subplot(325)
imshow(pc5, [])
title('Fifth Principal Component');

 
%--------------------------------------------------------------------------
% reconstructing 100th image with varying number of principal components
%--------------------------------------------------------------------------

% defining containers for 10, 100, 200, 399 component reconstruction
image_100_10 = zeros(64, 64);
image_100_100 = zeros(64, 64);
image_100_200 = zeros(64, 64);
image_100_399 = zeros(64, 64);

% reconstructing 100th image using 10 components
for i = 1:length(latent)
    component = (((coeff(:,i))*(coeff(:,i))')*(centred_image_100)');        % projecting data onto componenet dimension
    if i <= 10
        image_100_10 = image_100_10 + reshape(component', 64, 64);          % adding component to 10-reconstructed image
    end
    if i <= 100
        image_100_100 = image_100_100 + reshape(component', 64, 64);        % adding component to 100-reconstructed image
    end
    if i <= 200
        image_100_200 = image_100_200 + reshape(component', 64, 64);        % adding component to 200-reconstructed image
    end
    if i <= 399
        image_100_399 = image_100_399 + reshape(component', 64, 64);        % adding component to 399-reconstructed image
    end
end

% adding back the mean for display
image100_10 = image_100_10 + dataset_mean;
image100_100 = image_100_100 + dataset_mean;
image100_200 = image_100_200 + dataset_mean;
image100_399 = image_100_399 + dataset_mean;

% displaying reconstructed images
figure()
subplot(321)
imshow(image_100_matrix, [0 255])
title('Original 100th Image');
subplot(322)
imshow(image100_10, [0 255])
title('100th Image Reconstructed with 10 Principal Components');
subplot(323)
imshow(image100_100, [0 255])
title('100th Image Reconstructed with 100 Principal Components');
subplot(324)
imshow(image100_200, [0 255])
title('100th Image Reconstructed with 200 Principal Components');
subplot(325)
imshow(image100_399, [0 255])
title('100th Image Reconstructed with 399 Principal Components');
