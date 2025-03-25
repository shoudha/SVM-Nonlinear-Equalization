function errs = mmse_detector(y, signal_to_noise, bits)

noise_var = 1/signal_to_noise;
x_hat = tanh(y/noise_var);
bits_hat = (x_hat+1)/2;
errs = sum(bits_hat~=bits(1:length(bits_hat)))