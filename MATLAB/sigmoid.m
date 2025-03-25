clc
clearvars
close all

x = -10:.1:10;

y_1 = 1./(1+1*exp(-x));
y_m_10 = 1./(1+1*exp(-2*x));
y_10 = 1./(1+1*exp(-10*x));

hold on
plot(x,y_1, 'linewidth', 2)
plot(x,y_m_10, 'linewidth', 2)
plot(x,y_10, 'linewidth', 2)
hold off
grid on
legend('\gamma = 1', '\gamma = 2', '\gamma = 10')
title('Sigmoidal Activation Function')
xlabel('x')
title('Sigmoidal Activation Function: ${\it} (1+e^{-\gamma*x})^{-1}$','Interpreter','Latex')