%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generates the figure 5 in report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all

%%% Signal parameters
N = 1000; %Number of bits
h = [1 0.5]; % linear part of the channel
poly_coeff = [1 0 -0.9]; % Polynomial part of the channel
signal_to_noise = 15; %SNR
rho = 0; %Uncorrelated noise
D = 0; % Detector Delay = 0, 1, 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 5(a) Decision boundary for Polynomial Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 1
fprintf("Figure 5(a)\n")
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 5(b) Decision boundary for Gaussian Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 2
fprintf("Figure 5(a)\n")
kernel_type = 'gaussian'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 5(c) Decision boundary for Sigmoid Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 3
fprintf("Figure 5(c)\n")
gamma = 10; %Slope of sigmoid function
delta = 10; %Intercept of sigmoid function
kernel_type = 'sigmoid'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot', gamma, delta);
