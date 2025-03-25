clc
clearvars
close all

%Signal and channel parameters
N = 10000;
h = [1 0.5];
poly_coeff = [1 0 -.9];
signal_to_noise = 3:17;

D = 2;

nsims = length(signal_to_noise);
BER_SVM = zeros(1, nsims);
BER_ML = zeros(1, nsims);

parfor nsim = 1:nsims
    
    fprintf("Running for SNR: %d (%d/%d)\n", signal_to_noise(nsim), nsim, nsims)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Train an SVM model using these parameters 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % The train_svm_model trains an SVM model using the 
    % signal and channel parameters and saves the trianed
    % SVM model in SVMModels variable file.
    [u_train, X_train] = create_train_bayesian(N, h, D, poly_coeff, signal_to_noise(nsim));

    ML_errs = 0;
    
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
        y = awgn(x_hat, signal_to_noise(nsim), 'measured');

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

        sigma = 1/signal_to_noise(nsim);
        rho = 0.48;
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
        ML_errs = ML_errs + sum(bits_ml~=bits((1+D):(D+length(bits_ml))));
        
    
    BER_ML(nsim) = ML_errs/length(bits_ml);
    
end

BER_MLE_D_2 = BER_ML;
save('BER_MLE_D_2.mat','BER_MLE_D_2')

semilogy(signal_to_noise, BER_ML, '-s','linewidth', 2)

grid on
legend('ML Detector')
xlabel('Signal to Noise ratio')
ylabel('Bit Error Rate')

