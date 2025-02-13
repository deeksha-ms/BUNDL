function gauss_sig = generate_gauss(gaussons,gaussoffs, epochs, noiselevel)

if noiselevel==1
    noiseamp=0.5; 
elseif  noiselevel==2
    noiseamp = 0.65; 
elseif noiselevel==3
    noiseamp = 0.8; 
else
    noiseamp = 1.; 
end

gauss_noise = struct( ...
        'type', 'noise', ...
        'color', 'white', ...
        'amplitude', noiseamp);
gauss_noise = utl_check_class(gauss_noise);
tempsig = generate_signal_fromclass(gauss_noise, epochs);

s = [[0], gaussoffs]; 
e = [gaussons, [epochs.length/1000]]; 

for j=1:size(s, 2)
    tempsig(s(j)*epochs.srate+1:e(j)*epochs.srate) = 0;
end


gauss_sig = struct();
gauss_sig.data = tempsig;
gauss_sig.index = {'e', ':'};
gauss_sig.amplitude = 1.0;
gauss_sig.amplitudeType = 'relative';
gauss_sig = utl_check_class(gauss_sig, 'type', 'data');

end
