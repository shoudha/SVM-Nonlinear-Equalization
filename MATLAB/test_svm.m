clc
clearvars
close all

%Signal and channel parameters
N = 1000;
h = [1 0.5];
poly_coeff = [1 0 -0.9];
signal_to_noise = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Train an SVM model using these parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The train_svm_model trains an SVM model using the 
% signal and channel parameters and saves the trianed
% SVM model in SVMModels variable file.
train_svm_model(N, h, poly_coeff, signal_to_noise, 'noplot');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% TRANSMITTER SECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Generate random bits
bits = randi([0 1], N, 1);

%BPSK Modulation
u = 2*(bits-0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% CHANNEL SECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Convolution with channel coefficients
u_n = u(1:end-1);
u_n_1 = u(2:end);

x_tl = u_n.*h(1) + u_n_1.*h(2);

%Multiplication with channel polynomial coefficients
x_hat = zeros(size(x_tl));
for i = 1:length(x_tl)
    temp = 0;
    for j = 1:length(poly_coeff)
        temp = temp + poly_coeff(j)*x_tl(i)^j;
    end
    x_hat(i) = temp;
end

%Additive white gaussian noise
y = awgn(x_hat, signal_to_noise);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% RECEIVER SECTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load trained SVM model
load SVMModels
classes = size(SVMModels,1);

%Create SVM test variable
y_n = y(1:end-1);
y_n_1 = y(2:end);
X = [y_n y_n_1];

%Equalize with trained SVM model
Scores = zeros(size(X,1),classes);
for j = 1:classes
    [~,score] = predict(SVMModels{j},X);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);
bits_tl = maxScore - 1;

%BER calculation
errs = sum(bits_tl~=bits(1:end-2));
BER = errs/length(bits_tl)





