% developed by Riley Black

% initializing MATLAB environment
close all;
clc;
clear all;

% loading image dataset into workspace
dataset = importdata('faces.dat');


%--------------------------------------------------------------------------
% visualizing dataset
%--------------------------------------------------------------------------

% computing mean-centred image dataset
dataset_mean = mean2(dataset);
centred_dataset = dataset - dataset_mean;

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

% plotting eigenvalues of the covariance matrix of the centred dataset
figure()
plot(latent)
xlim([-50 450])
ylim([-200000 1200000])
grid on;
title('Eigenvalues');
xlabel('Order');
ylabel('Magnitude');


%--------------------------------------------------------------------------
% analyzing data dimensionality
%--------------------------------------------------------------------------

% computing total varience 
total_varience = sum(latent);

% defining variable tracking varience captured by i first eigenvalues
running_varience = 0;

% defining container to store tracking results 
running_varience_history = zeros(400,1);

% defining variable to hold id of eigenvalue cut off id
cutoff = 0;

% variable to reflect if the cutoff has been found yet 
foundcutoff = 0;

for i = 1:399 % for each eigenvalue
    running_varience = running_varience + latent(i,1); % update new accounted for varience total
    running_varience_history(i+1,1) = running_varience/total_varience; % track proportion accounted for
    if (running_varience_history(i+1,1) > 0.95) && (foundcutoff == 0) % if accounted for proportion is greater that 95% and havent already found cutoff
        foundcutoff = 1; % update flag to reflect found cutoff
        cutoff = i; % save eigenvalue number
    end
end

% plot First d Eigenvalues Proportion of Variance Captured
figure(4)
plot(running_varience_history)
xlim([-50 450])
ylim([-0.1 1.1])
grid on;
title('First d Eigenvalues Proportion of Varience Captured');
xlabel('d');
ylabel('Proportion of Varience Captured');
% 
% 
% %----------------------
% % PART F
% %----------------------
% 
% % reshape first eigenvector into 64x64 matrix
% eigenmat1 = reshape(coeff(:,1),64,64);
% % displaying image
% figure(5)
% imshow(eigenmat1, [])
% 
% % reshape second eigenvector into 64x64 matrix
% eigenmat2 = reshape(coeff(:,2),64,64);
% % displaying image
% figure(6)
% imshow(eigenmat2, [])
% 
% % reshape third eigenvector into 64x64 matrix
% eigenmat3 = reshape(coeff(:,3),64,64);
% % displaying image
% figure(7)
% imshow(eigenmat3, [])
% 
% % reshape fourth eigenvector into 64x64 matrix
% eigenmat4 = reshape(coeff(:,4),64,64);
% % displaying image
% figure(8)
% imshow(eigenmat4, [])
% 
% % reshape fifth eigenvector into 64x64 matrix
% eigenmat5 = reshape(coeff(:,5),64,64);
% % displaying image
% figure(9)
% imshow(eigenmat5, [])
% 
% 
% %----------------------
% % PART G
% %----------------------
% 
% % variable to hold reconstructed image using 10 components
% xbar1 = zeros(64,64);
% 
% for i = 1:10 % for each of the first 10 components
%     xbari1 = (((coeff(:,i))*(coeff(:,i))')*(image_100_mean_centred)'); % equation given in assignment
%     xbar1 = xbar1 + reshape(xbari1',64,64); % sum kth element contribution to running total
% end
% 
% xbarorig1 = xbar1 + datamean; % add back the mean to reconstruct
% 
% % display image
% figure(10)
% imshow(xbarorig1, [0 255])
% 
% 
% 
% 
% 
% % variable to hold reconstructed image using 100 components
% xbar2 = zeros(64,64);
% 
% for i = 1:100 % for each of the first 100 components
%     xbari2 = (((coeff(:,i))*(coeff(:,i))')*(image_100_mean_centred)'); % equation given in assignment
%     xbar2 = xbar2 + reshape(xbari2',64,64); % sum kth element contribution to running total
% end
% 
% xbarorig2 = xbar2 + datamean; % add back the mean to reconstruct
% 
% % display image
% figure(11)
% imshow(xbarorig2, [0 255])
% 
% 
% 
% 
% 
% 
% % variable to hold reconstructed image using 200 components
% xbar3 = zeros(64,64);
% 
% for i = 1:200  % for each of the first 200 components
%     xbari3 = (((coeff(:,i))*(coeff(:,i))')*(image_100_mean_centred)'); % equation given in assignment
%     xbar3 = xbar3 + reshape(xbari3',64,64); % sum kth element contribution to running total
% end
% 
% xbarorig3 = xbar3 + datamean; % add back the mean to reconstruct
% 
% % display image
% figure(12)
% imshow(xbarorig3, [0 255])
% 
% 
% 
% 
% 
% % variable to hold reconstructed image using 399 components
% xbar4 = zeros(64,64);
% 
% for i = 1:399  % for each of the first 399 components
%     xbari4 = (((coeff(:,i))*(coeff(:,i))')*(image_100_mean_centred)'); % equation given in assignment
%     xbar4 = xbar4 + reshape(xbari4',64,64); % sum kth element contribution to running total
% end
% 
% xbarorig4 = xbar4 + datamean; % add back the mean to reconstruct
% 
% % display image
% figure(13)
% imshow(xbarorig4, [0 255])