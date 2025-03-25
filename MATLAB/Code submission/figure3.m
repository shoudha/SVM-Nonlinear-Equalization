%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generates the figure 3 in report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all

%%% Signal parameters
N = 1000; %Number of bits
h = [1 0.5]; % linear part of the channel
poly_coeff = [1 0 -0.9]; % Polynomial part of the channel
signal_to_noise = 16; %SNR
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 3(a) Decision boundary for D = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 1
fprintf("Figure 3(a)\n")
D = 0;
rho = 0; %uncorrelated noise
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 3(b) Decision boundary for D = 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 2
fprintf("Figure 3(b)\n")
D = 1;
rho = 0; %uncorrelated noise
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 3(c) Decision boundary for D = 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 3
fprintf("Figure 3(c)\n")
D = 2;
rho = 0; %uncorrelated noise
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 3(d) Decision boundary for 
%%% Colored noise and D = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 4
fprintf("Figure 3(d)\n")
D = 0;
rho = 0.48; %correlated noise with correlation coefficient rho = 0.48
train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise, kernel_type, 'plot');
