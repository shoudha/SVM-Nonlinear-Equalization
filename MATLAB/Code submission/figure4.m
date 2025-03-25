%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generates the figure 4 in report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all

%%% Signal parameters
N = 10000; %Number of bits
h = [1 0.5]; % linear part of the channel
poly_coeff = [1 0 -0.9]; % Polynomial part of the channel
signal_to_noise = 3:17; %SNR
rho = 0; %Uncorrelated noise

niter = 1000; %Iteration for averaging over BER
nsims = length(signal_to_noise);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 4(a) 
%%% BER vs SNR for different Detector Delays D = 0,1,2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 1
fprintf("Figure 4(a)\n")
fprintf("D = 0\n")
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

%For D = 0
D = 0;
BER_D0 = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_D0(nsim) = SVM_errs;    
end

%For D = 1
fprintf("D = 1\n")
D = 1;
BER_D1 = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_D1(nsim) = SVM_errs;    
end

%For D = 2
fprintf("D = 2\n")
D = 2;
BER_D2 = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_D2(nsim) = SVM_errs;    
end

%Plot
figure
semilogy(signal_to_noise, BER_D0, '-s',...
        signal_to_noise, BER_D1, '-s',...
        signal_to_noise, BER_D2, '-s',...
         'linewidth', 2)
grid on
legend( 'D = 0',...
        'D = 1',...
        'D = 2')
xlabel('Signal to Noise ratio')
ylabel('Bit Error Rate')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 4(b) 
%%% BER vs SNR for AWGN, Colored noise and optimum MLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 2
fprintf("Figure 4(b)\n")
fprintf("AWGN...\n")

%For AWGM
D = 0;
BER_AWGN = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_AWGN(nsim) = SVM_errs;    
end

%For Colored Noise
fprintf("Colored Noise...\n")
BER_colored = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    % For colored noise, rho = 0.48
    SVMModels = train_svm_model(N, h, D, poly_coeff, 0.48, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        % For colored noise, rho = 0.48
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, 0.48, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_colored(nsim) = SVM_errs;    
end

%For Optimal MLE
fprintf("Optimal MLE...\n")
BER_ML = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Create training variable for Bayesian MLE optimum detector
    [u_train, X_train] = train_bayesian(N, h, D, poly_coeff, rho, signal_to_noise(nsim));
    
    %Test Bayesian model
    BER_ML(nsim) = test_bayesian(u_train, X_train, N, h, D, poly_coeff, rho, signal_to_noise(nsim));
end

%Plot
semilogy(signal_to_noise, BER_AWGN, '-s',...
        signal_to_noise, BER_colored, '-s',...
        signal_to_noise, BER_ML, '-s',...
         'linewidth', 2)
grid on
figure
legend( 'SVM Detector - AWGN',...
        'SVM Detector - Colored noise',...
        'SVM Detector - Optimal MLE')
xlabel('Signal to Noise ratio')
ylabel('Bit Error Rate')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 4(c) 
%%% BER vs SNR for different SVM kernels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Figure 3
fprintf("Figure 4(c)\n")
D = 0;

%For kernel = polynomial
fprintf("Kernel = Polynomial\n")
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

BER_poly = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_poly(nsim) = SVM_errs;    
end

%For kernel = Gaussian or RBF
fprintf("Kernel = Gaussian\n")
kernel_type = 'gaussian'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

BER_rbf = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot');
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_rbf(nsim) = SVM_errs;    
end

%For kernel = Sigmoid
fprintf("Kernel = Sigmoid\n")
kernel_type = 'sigmoid'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'
gamma = 10;
delta = 10;

BER_sig = zeros(1, nsims);
%Train and test SVM models at each SNR
for nsim = 1:nsims
    
    %Train SVM model
    SVMModels = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise(nsim), kernel_type, 'noplot', gamma, delta);
    
    %Test SVM model
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise(nsim), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_sig(nsim) = SVM_errs;    
end

semilogy(signal_to_noise, BER_poly, '-s',...
        signal_to_noise, BER_rbf, '-s',...
        signal_to_noise, BER_sig, '-s',...
         'linewidth', 2)
grid on
figure
legend( 'SVM Kernel - Polynomial',...
        'SVM Kernel - Gaussian (RBF)',...
        'SVM Kernel - Sigmoid')
xlabel('Signal to Noise ratio')
ylabel('Bit Error Rate')



























