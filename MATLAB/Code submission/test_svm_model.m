function err_rate = test_svm_model(SVMModels, N, h, D, poly_coeff, rho, signal_to_noise, plot)
% test_svm_model(SVMModels, N, h, D, poly_coeff_train1, rho, signal_to_noise, plot)
% tests a trained SVM model 'SVMModels'.
% N = number of bits,
% h = channel coefficients,
% D = detector delay,
% poly_coeff = channel polynomial coefficients,
% rho = noise correlation coefficient,
% signal_to_noise = Signal to noise ratio,
% kernel_type = 'polynomial', 'gaussian', 'sigmoid',
% plot = 'plot' for generating decision boundaries, 'noplot' to skip
% plotting decision boundaries.
%
% Output = err_rate is the number of errors divided by bit sequence length 
%          after equalized using the trained SVM model

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
    y = awgn(x_hat, signal_to_noise, 'measured');
end      

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% RECEIVER SECTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% SVM Detector         
%Load trained SVM model
classes = size(SVMModels,1);

%Create SVM test variable
y_n = y(1:end-1);
y_n_1 = y(2:end);
X = [y_n y_n_1];

fprintf('Testing SVM model...\n')
fprintf('\n')

%Equalize with trained SVM model
Scores = zeros(size(X,1),classes);
for j = 1:classes
    [~,score] = predict(SVMModels{j},X);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);
bits_tl = maxScore - 1;

%Error calculation
err_rate = sum(bits_tl~=bits((1+D):(D+length(bits_tl))))/length(bits_tl);

if strcmp(plot, 'plot')
    
    %Prepare variables for plot
    Y_n = [y_n y_n_1];
    Y_n_pos = Y_n(u(1+D:end+(D-2))==1,:);
    Y_n_neg = Y_n(u(1+D:end+(D-2))==-1,:);
    X = [Y_n_pos; Y_n_neg];
    Y = [repmat({'1'}, size(Y_n_pos,1),1); repmat({'-1'}, size(Y_n_neg,1),1)];

    d = 0.008;
    [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
        min(X(:,2)):d:max(X(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    N = size(xGrid,1);
    Scores = zeros(N,2);

    for j = 1:2
        [~,score] = predict(SVMModels{j},xGrid);
        Scores(:,j) = score(:,2); % Second column contains positive-class scores
    end

    [~,maxScore] = max(Scores,[],2);

    %Save the trained SVM model
    figure
    h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
                [1 .4 .4; .4 .4 1]);
    hold on
    h(3:4) = gscatter(X(:,1),X(:,2),Y,'br','xo',5);
    xlabel('x(n)');
    ylabel('x(n-1)');
    legend(h,{  '+1 region','-1 region',...
                '+1','-1'},...
                'Location','Northwest');
    axis tight
    hold off
end
