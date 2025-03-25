function [u_mod, Y_n] = create_train_bayesian(N, h, D, poly_coeff, signal_to_noise)

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
y = awgn(x_hat, signal_to_noise, 'measured');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% RECEIVER 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Prepare training variables for SVM
u_mod = u(1+D:end+(D-2));
y_n = y(1:end-1);
y_n_1 = y(2:end);
Y_n = [y_n y_n_1];
Y_n_pos = Y_n(u_mod==1,:);
Y_n_neg = Y_n(u_mod==-1,:);
