function bgsignal = generate_background(ons, offs, epochs)
    %%% please input ons and offs of size 3 [on:off] is where bg is

    bgsignal  = struct( ...
            'type', 'ersp', ...
            'frequency', [8 10 15 28], ...
            'amplitude', 0.85, ...
            'modulation', 'none');
    bgsignal = utl_check_class(bgsignal);
    
    if size(ons)>0
        tempsignal = generate_signal_fromclass(bgsignal, epochs); 
        tempsignal(ons(1)*epochs.srate-1: offs(1)*epochs.srate) = 0;
        
        bgsignal = struct();
        bgsignal.data = tempsignal;
        bgsignal.index = {'e', ':'};
        bgsignal.amplitude = 1.0;
        bgsignal.amplitudeType = 'relative';
        bgsignal = utl_check_class(bgsignal, 'type', 'data');
    end
end