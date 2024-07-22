function slowsignal = generate_slow(slowstart,slowend, epochs)
    
    slowsignal  = struct( ...
        'type', 'ersp', ...
        'frequency', [1 2 2.5 3], ...
        'amplitude', 1.0, ...
        'modulation', 'none');

    slowsignal = utl_check_class(slowsignal);
    
    tempsig = generate_signal_fromclass(slowsignal, epochs);
    tempsig(1:slowstart*epochs.srate ) = 0;
    tempsig(slowend*epochs.srate-1:end ) = 0;
    slowsignal = struct();
    slowsignal.data = tempsig;
    slowsignal.index = {'e', ':'};
    slowsignal.amplitude = 1.0;
    slowsignal.amplitudeType = 'relative';
    slowsignal = utl_check_class(slowsignal, 'type', 'data');
    
end