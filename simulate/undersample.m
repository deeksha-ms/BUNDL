function noisylabel = undersample(truelabel, noiselevel)

    delta = truelabel(2:end) - truelabel(1:end-1);
    onset = find(delta==1)+1;
    if isempty(onset)
        onset = 1;
    end

    offset = find(delta==-1)+1;
    if isempty(offset)
        offset = 600;
    end
    szlen = offset-onset+1;
    samples_to_remove = round(noiselevel*(szlen));
    
    noise_onset = min(offset-5, onset+samples_to_remove);
    noisylabel = truelabel;
    noisylabel(onset:noise_onset-1) = 0.;
    
end