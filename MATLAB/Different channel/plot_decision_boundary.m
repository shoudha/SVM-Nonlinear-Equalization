function plot_decision_boundary(y, u, SVMModels)

%Create SVM test variable
y_n = y(1:end-1);
y_n_1 = y(2:end);
D=0;
Y_n = [y_n y_n_1];
Y_n_pos = Y_n(u(1+D:end+(D-2))==1,:);
Y_n_neg = Y_n(u(1+D:end+(D-2))==-1,:);
X = [Y_n_pos; Y_n_neg];
Y = [repmat({'1'}, size(Y_n_pos,1),1); repmat({'-1'}, size(Y_n_neg,1),1)];

%Equalize with trained SVM model
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



