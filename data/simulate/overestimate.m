function noisylabel = overestimate(truelabel, noiselevel)

    delta = truelabel(2:end) - truelabel(1:end-1);
    onset = find(delta==1)+1;
    if isempty(onset)
        onset = 1;
    end

    offset = find(delta==-1)+1;
    if isempty(offset)
        offset = 600;
    end

    samples_to_add = round(noiselevel*60);
    
    noise_onset = max(1, onset-samples_to_add);
    noisylabel = truelabel;
    noisylabel(noise_onset:onset) = 1.;
    
    rem =  samples_to_add - (sum(noisylabel) - sum(truelabel));
    
    if rem>0
        noisyoffset = min(600, offset+rem-1);
         noisylabel(offset:noisyoffset) = 1.;
    end

end