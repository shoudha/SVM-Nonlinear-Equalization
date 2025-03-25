function err_rate = test_bayesian(u_train, X_train, N, h, D, poly_coeff, rho, signal_to_noise)
% err_rate = test_bayesian(u_train, X_train, N, h, D, poly_coeff, rho, signal_to_noise)
% tests the optimal Bayesian Maximum Liklihood detector.
% u_train = bit sequence created by train_bayesian,
% X_train = symbol pairs corresponding to u_train bit sequence,
% N = number of bits,
% h = channel coefficients,
% D = detector delay,
% poly_coeff = channel polynomial coefficients,
% rho = noise correlation coefficient,
% signal_to_noise = SNR,
%
% Output = err_rate, number of errors divided by bit sequence length.

fprintf("Testing bayesian MLE...\n")

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
y = awgn(x_hat, signal_to_noise, 'measured');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% RECEIVER SECTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Create test variable
y_n = y(1:end-1);
y_n_1 = y(2:end);
X = [y_n y_n_1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% BAYESIAN SECTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot(X_train(u_train==1,1),X_train(u_train==1,2),'x')
% hold on
% plot(X_train(u_train==-1,1),X_train(u_train==-1,2),'o')
% hold off

H1 = zeros(size(X,1),1);
H0 = zeros(size(X,1),1);

X_train_pos = X_train(u_train==1,:);
X_train_neg = X_train(u_train==-1,:);

sigma = 1/signal_to_noise;
E = sigma^2*[1 rho;rho 1];
E_inv = pinv(E);

for i = 1:size(X,1)
    o = X(i,:)';
    for j = 1:size(X_train_pos,1)
        o_pos_hat = X_train_pos(j,:)';
        exp_pos = -0.5*(o-o_pos_hat)'*E_inv*(o-o_pos_hat);
        H1(i) = H1(i) + exp(exp_pos);
    end
    for j = 1:size(X_train_neg,1)
        o_neg_hat = X_train_neg(j,:)';
        exp_neg = -0.5*(o-o_neg_hat)'*E_inv*(o-o_neg_hat);
        H0(i) = H0(i) + exp(exp_neg);
    end
end

bits_ml = H1>H0;

%BER calculation
err_rate = sum(bits_ml~=bits((1+D):(D+length(bits_ml))))/length(bits_ml);





