%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Implements the SVM bank from 
%%% figure 7 in the report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clearvars
close all

%%% Signal parameters
N = 10000; %Number of bits
h = [1 0.5]; % linear part of the channel
poly_coeff = [1 0 -0.9]; % Polynomial part of the channel

niter = 1000; %Iteration for averaging over BER
rho = 0; %Uncorrelated noise
D = 0; % Detector Delay = 0, 1, 2
kernel_type = 'polynomial'; %SVM kernel Types = 'polynomial', 'gaussian', 'sigmoid'

% Training and testing SNR
signal_to_noise_train = 3:17;
signal_to_noise_test = 3:17;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Train one SVM model for each SNR value in training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntrains = length(signal_to_noise_train);
SVMBank = cell(ntrains,1); %Cell array to store the trained SVM models

for ntrain = 1:ntrains
    
    fprintf("Running for SNR: %d (%d/%d)\n", signal_to_noise_train(ntrain), ntrain, ntrains)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% Train an SVM model using these parameters 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % The train_svm_model trains an SVM model using the 
    % signal and channel parameters and saves the trianed
    % SVM model in SVMModels variable file.
    SVMBank{ntrain} = train_svm_model(N, h, D, poly_coeff, rho, signal_to_noise_train(ntrain), kernel_type, 'noplot');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Test the SVM models using test SNR dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntests = length(signal_to_noise_test);
BER_svm_bank = zeros(1, ntests);

for ntest = 1:ntests
    
    %Test SVM model
    SVMModels = SVMBank{ntest}; % Select the proper trained SVM model based on received SNR
    SVM_errs = 0;
    for iter = 1:niter
        err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise_test(ntest), 'noplot');
        SVM_errs = SVM_errs + err_rate/niter;
    end
    BER_svm_bank(ntest) = SVM_errs; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Calculate optimum MLE for comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%For Optimal MLE
BER_ML = zeros(1, ntests);
%Train and test SVM models at each SNR
for ntest = 1:ntests
    
    %Create training variable for Bayesian MLE optimum detector
    [u_train, X_train] = train_bayesian(N, h, D, poly_coeff, rho, signal_to_noise_test(ntest));
    
    %Test Bayesian model
    BER_ML(ntest) = test_bayesian(u_train, X_train, N, h, D, poly_coeff, rho, signal_to_noise_test(ntest));
end

%Plot
semilogy(   signal_to_noise_test, BER_svm_bank, '-s',...
            signal_to_noise_test, BER_ML, '-s',...
         'linewidth', 2)
grid on
legend('SVM Bank Detector', 'Optimal Detector')
xlabel('Signal to Noise ratio')
ylabel('Bit Error Rate')

