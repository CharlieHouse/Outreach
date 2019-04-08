clear all; close all;

load('plant.mat')
t = linspace(0,length(mic)/48000,length(mic));

% Tidy Up & Remove Latency
mic = mic-mean(mic);

[r,lag] = xcorr(noise,mic);
[~,ind] = max(r);
lag = lag(ind);
mic = mic(-lag+1:end);
noise = noise(1:length(mic));
% 
wind = window(@flattopwin,length(mic));
plot(wind)
mic = mic .* wind.';
noise = noise.*wind.';

%% Calculate TF
nfft=2^14;
noverlap=nfft/2;
window=nfft;
fs = 48000;

sxx = pwelch(noise,window,noverlap,nfft,fs);
[sxy,f] = cpsd(noise,mic,window,noverlap,nfft,fs);
H = sxy./sxx;
coh = mscohere(noise,mic,window,noverlap,nfft,fs);

figure()
plot(f,coh)
figure()
plot(f,20*log10(abs(H)))