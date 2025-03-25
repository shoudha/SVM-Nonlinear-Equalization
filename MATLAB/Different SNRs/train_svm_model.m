function train_svm_model(N, h, poly_coeff, signal_to_noise, plot)

% train_svm_model(N, h, poly_coeff, signal_to_noise)
% trains an SVM model saves the trained SVM model in SVMModels.
% N = number of bits,
% h = channel coefficients,
% poly_coeff = channel polynomial coefficients,
% signal_to_noise = Signal to noise ratio.
% plot = 'plot' for generating decision boundaries.

X = [];
Y = [];

ntrains = length(signal_to_noise);
for ntrain = 1:ntrains

    fprintf("Generating training data for SNR: %d (%d/%d)\n", signal_to_noise(ntrain), ntrain, ntrains)
    
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
    y = awgn(x_hat, signal_to_noise(ntrain));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%% RECEIVER 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %Prepare training variables for SVM
    D = 0;
    y_n = y(1:end-1);
    y_n_1 = y(2:end);
    Y_n = [y_n y_n_1];
    Y_n_pos = Y_n(u(1+D:end+(D-2))==1,:);
    Y_n_neg = Y_n(u(1+D:end+(D-2))==-1,:);

    X = [X; Y_n_pos; Y_n_neg];
    Y = [Y; repmat({'1'}, size(Y_n_pos,1),1); repmat({'-1'}, size(Y_n_neg,1),1)];

end

fprintf("Training SVM...\n")

classes = unique(Y);
SVMModels = cell(numel(classes),1);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = strcmp(classes(j),Y); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint', 3);
end

save('SVMModels.mat','SVMModels')

%predict the output using trained SVM
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

%Save the trained SVM model
if strcmp(plot, 'plot')
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



