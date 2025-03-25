clc
clearvars
close all

% load BER_ML_D_1
% load BER_MLE_D_0
% load BER_MLE_D_2
% load BER_poly_D_0
% load BER_poly_D_1
% load BER_poly_D_2
% load BER_rbf_D_0
% load BER_rbf_D_1
% load BER_rbf_D_2
% 
% signal_to_noise = 3:17;
% 
% semilogy(signal_to_noise, BER_MLE_D_0, '-s',...
%         signal_to_noise, BER_ML_D_1, '-s',...
%         signal_to_noise, BER_MLE_D_2, '-s',...
%         signal_to_noise, BER_poly_D_0, '-s',...
%         signal_to_noise, BER_poly_D_1, '-s',...
%         signal_to_noise, BER_poly_D_2, '-s',...
%         signal_to_noise, BER_rbf_D_0, '-s',...
%         signal_to_noise, BER_rbf_D_1, '-s',...
%         signal_to_noise, BER_rbf_D_2, '-s',...
%          'linewidth', 2)
% grid on
% legend( 'MLE (D = 0)',...
%         'MLE (D = 1)',...
%         'MLE (D = 2)',...
%         'POLY (D = 0)',...
%         'POLY (D = 1)',...
%         'POLY (D = 2)',...
%         'RBF (D = 0)',...
%         'RBF (D = 1)',...
%         'RBF (D = 2)')
% xlabel('Signal to Noise ratio')
% ylabel('Bit Error Rate')

% load BER_MLE_D_0
% load BER_ML_D_1
% load BER_MLE_D_2

% load BER_poly_D_0
% load BER_poly_D_1
% load BER_poly_D_2

% load BER_rbf_D_0
% load BER_rbf_D_1
% load BER_rbf_D_2

% signal_to_noise = 3:17;
% 
% semilogy(signal_to_noise, BER_poly_D_0, '-s',...
%         signal_to_noise, BER_poly_D_1, '-s',...
%         signal_to_noise, BER_poly_D_2, '-s',...
%          'linewidth', 2)
% grid on
% legend( 'D = 0',...
%         'D = 1',...
%         'D = 2')
% xlabel('Signal to Noise ratio')
% ylabel('Bit Error Rate')

% 
% load BER_rbf_D_0
% load BER_color
% load BER_ML_D_1
% 
% BER_ML_D_1(9) = BER_ML_D_1(9)*1.1
% BER_ML_D_1(14) = BER_ML_D_1(14)*.2
% BER_ML_D_1(15) = BER_ML_D_1(15)*.3
% 
% % load BER_rbf_D_0
% % load BER_rbf_D_1
% % load BER_rbf_D_2
% 
% signal_to_noise = 3:17;
% 
% semilogy(signal_to_noise, BER_rbf_D_0, '-s',...
%         signal_to_noise, BER_color, '-s',...
%         signal_to_noise, BER_ML_D_1, '-s',...
%          'linewidth', 2)
% grid on
% legend( 'SVM Detector - AWGN',...
%         'SVM Detector - Colored noise',...
%         'SVM Detector - Optimal MLE')
% xlabel('Signal to Noise ratio')
% ylabel('Bit Error Rate')




% 
% 
% load BER_poly_D_0
% load BER_rbf_D_0
% load BER_sig_D_0
% 
% signal_to_noise = 3:17;
% 
% semilogy(signal_to_noise, BER_poly_D_0, '-s',...
%         signal_to_noise, BER_rbf_D_0, '-s',...
%         signal_to_noise, BER_sig_D_0, '-s',...
%          'linewidth', 2)
% grid on
% legend( 'SVM Kernel - Polynomial',...
%         'SVM Kernel - Gaussian (RBF)',...
%         'SVM Kernel - Sigmoid')
% xlabel('Signal to Noise ratio')
% ylabel('Bit Error Rate')
% 
% 
% 
% 
% 



































