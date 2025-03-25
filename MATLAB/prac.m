clc
clearvars
% close all

N = 1000;
u = 2*(randi([0 1], N, 1)-0.5);
h = [1 0.5];
poly_coeff = [1 0 -0.9];
signal_to_noise = 12;
noise_var = .2;
D = 0;

u_n = u(1:end-1);
u_n_1 = u(2:end);

x_tl = u_n.*h(1) + u_n_1.*h(2);

x_hat = zeros(size(x_tl));
for i = 1:length(x_tl)
    temp = 0;
    for j = 1:length(poly_coeff)
        temp = temp + poly_coeff(j)*x_tl(i)^j;
    end
    x_hat(i) = temp;
end

y = awgn(x_hat, signal_to_noise);

y_n = y(1:end-1);
y_n_1 = y(2:end);

Y_n = [y_n y_n_1];
Y_n_pos = Y_n(u(1+D:end+(D-2))==1,:);
Y_n_neg = Y_n(u(1+D:end+(D-2))==-1,:);

X = [Y_n_pos; Y_n_neg];
Y = [ones(size(Y_n_pos,1),1); -ones(size(Y_n_neg,1),1)];

cla;
hold on
scatter(Y_n_pos(:,1), Y_n_pos(:,2), 'ro', 'LineWidth', 2)
scatter(Y_n_neg(:,1), Y_n_neg(:,2), 'bx', 'LineWidth', 2)
hold off
grid on
axis([-2.5 2.5 -2.5 2.5])
xlabel('x(n)')
ylabel('x(n-1)')




