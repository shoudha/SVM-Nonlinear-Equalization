function [u_mod, Y_n] = train_bayesian(N, h, D, poly_coeff, rho, signal_to_noise)
% [u_mod, Y_n] = train_bayesian(N, h, D, poly_coeff, rho, signal_to_noise)
% creates training data points that are used for an Optimal one/zero cost Maximum Likelihood Detector.
% N = number of bits,
% h = channel coefficients,
% D = detector delay,
% poly_coeff = channel polynomial coefficients,
% rho = noise correlation coefficient,
% signal_to_noise = SNR,
%
% Output => u_mod is the bit sequence for reference
%        => Y_n is the symbol pairs for detector.

fprintf("Creating training variable for bayesian MLE...\n")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% TRANSMITTER 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Generate random bits
bits = randi([0 1], N, 1);

%BPSK Modulation
u = 2*(bits-0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% CHANNEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
if rho ~= 0

    sigPower = sum(x_hat.^2)./length(x_hat);    
    noisePower = sigPower/(10^(signal_to_noise/10));
    noiseWGN = randn(length(x_hat),1);

    for jj = 1:1:length(x_hat)-1
       x_hat_tmp = x_hat(jj:jj+1);
       noiseVec = noiseWGN(jj:jj+1);%randn(2,1); 
       cMatrix = [1 rho;rho 1];%noise_var*[1 rho;rho 1];
       noise = sqrt(noisePower)*cMatrix*noiseVec;
       x_hat_tmp = x_hat_tmp + noise;
       x_hat(jj) = x_hat_tmp(1);
    end
    y = x_hat;
else 
    y = awgn(x_hat, signal_to_noise,'measured');
end   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% RECEIVER 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Prepare training variables for SVM
u_mod = u(1+D:end+(D-2));
y_n = y(1:end-1);
y_n_1 = y(2:end);
Y_n = [y_n y_n_1];
