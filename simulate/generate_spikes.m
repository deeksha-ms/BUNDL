function sig = generate_spikes(spikeons, spikeoffs, epochs)
    signal = zeros(1, epochs.srate*epochs.length/1000); 
    
    for j=1:size(spikeons, 2)
        
        len = sum(spikeoffs(j)-spikeons(j)); 
        minspikes = 6*len; 
        maxspikes = 14*len; 
        spikedata = erp_get_class_random([minspikes:maxspikes], ...  %number of spikes range
                                         [spikeons(j)*1e03:spikeoffs(j)*1e03], ... %given timepoints only
                                         [10:50], ...                 %width of spike rang
                                          [-1:0.05:-0.9, 0.9:0.05:1]);  %amplitude range
        signal = signal + generate_signal_fromclass(spikedata, epochs); 

    end

    sig = struct();
    sig.data = signal;
    sig.index = {'e', ':'};
    sig.amplitude = 1.0;
    sig.amplitudeType = 'relative';
    sig = utl_check_class(sig, 'type', 'data');
end
