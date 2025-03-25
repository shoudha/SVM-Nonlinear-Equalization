%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generates the figure 6 in report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all

%%% Signal parameters
N = 1000; %Number of bits
signal_to_noise = 15; %SNR
rho = 0; %Uncorrelated noise
D = 0; % Detector Delay = 0, 1, 2
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

%First training and testing channel set
h_train1 = [1 0.9]; % 1st training channel linear part
poly_coeff_train1 = [1 0 -0.9]; % 1st training channel polynomial part
h_test1 = [1 0.5]; % 1st training channel linear part
poly_coeff_test1 = [1 0 -0.9]; % 1st training channel polynomial part

%Second training and testing channel set
h_train2 = [1 0.6]; % 2nd training channel linear part
poly_coeff_train2 = [1 0 -0.5]; % 2nd training channel polynomial part
h_test2 = [1 0.6]; % 2nd testing channel linear part
poly_coeff_test2 = [1 0 -0.3]; % 2nd testing channel polynomial part

%Training SNR values
signal_to_noise_train = 3:17;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 6(a) Decision boundary for first training and
%%% testing set channels.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train SVM model using h_train1, poly_coeff_train1
% and save the trained model.
fprintf("Figure 6(a)\n")
SVMModels1 = train_svm_model(N, h_train1, D, poly_coeff_train1, rho, signal_to_noise, kernel_type, 'noplot');

% Test the trained SVM model using h_test1, poly_coeff_test1
test_svm_model(SVMModels1, N, h_test1, D, poly_coeff_test1, rho, signal_to_noise, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 6(b) Decision boundary for first training and
%%% testing set channels.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train SVM model using h_train2, poly_coeff_train2
% and save the trained model.
fprintf("Figure 6(b)\n")
SVMModels1 = train_svm_model(N, h_train2, D, poly_coeff_train2, rho, signal_to_noise, kernel_type, 'noplot');

% Test the trained SVM model using h_test2, poly_coeff_test2
test_svm_model(SVMModels1, N, h_test2, D, poly_coeff_test2, rho, signal_to_noise, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 6(c) Decision boundary for first training and
%%% testing set channels.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train and generate plot using the trained SVM model.
% The SVM model here is trained using signal_to_noise_train
fprintf("Figure 6(c)\n")
N = 300; %Low number bits makes it faster to train the SVM
train_svm_model(N, h_train2, D, poly_coeff_train2, rho, signal_to_noise_train, kernel_type, 'plot');

