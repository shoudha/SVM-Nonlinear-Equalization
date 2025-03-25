clc
clearvars
close all

N = 1000;
u = 2*(randi([0 1], N, 1)-0.5);
h = [1 0.5];
poly_coeff = [1 0 -0.9];
signal_to_noise = 12;
D = 0;




%SVM TRAIN
X = [Y_n_pos; Y_n_neg];
Y = [repmat({'1'}, size(Y_n_pos,1),1); repmat({'-1'}, size(Y_n_neg,1),1)];

classes = unique(Y);

indx = strcmp(classes(1),Y); % Create binary classes for each classifier
SVMModel = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
    'KernelFunction', 'polynomial', 'PolynomialOrder',3,'BoxConstraint', 3);

%PLOT
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

[~,score] = predict(SVMModel,xGrid);
Scores(:,j) = score(:,2); % Second column contains positive-class scores

[~,maxScore] = max(Scores,[],2);

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

%BER Calculation


