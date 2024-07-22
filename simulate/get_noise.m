function [s1, s2] = get_noise(sz_on, sz_end, sourcefield, e, noiselevel)
if noiselevel == 1
    amp_range = [-1.1:0.05:-0.9, 0.9:0.05:1.1]; 
    amp = 1.;
elseif noiselevel == 2
    amp_range = [-0.9:0.05:-0.75, 0.75:0.05:0.9];
    amp = 0.8;
elseif noiselevel ==3
    amp_range = [-0.75:0.05:-0.6, 0.6:0.05:0.75];
    amp = 0.7;
else 
    amp_range = [-0.6:0.05:-0.45, 0.45:0.05:0.6];
    amp = 0.6; 
end


    max_spikes = round(sz_end - sz_on +1)*100;
    %beg of seizure
    time_on1 = max(0, sz_on-30); 
    time_off1 = min(sz_on+30, 600); 
    art_on1 = time_on1 + randi(45); 
    art_off1 = art_on1 + randi([1,10]);
    
    %end of seizure
    time_on2 = sz_end; 
    time_off2 = min(sz_end+60, 600); 
    art_on2 = time_on2 + randi(45); 
    art_off2 = art_on2 + randi(20);
    
    s1 = erp_get_class_random([max_spikes-20:max_spikes], ...  %number of spikes range
                                     [art_on1*1e03:(art_off1)*1e03, art_on2*1e03:(art_off2)*1e03], ... %given timerange only
                                     [20:50], ...                 %width of spike range
		                             amp_range);  %amplitude range
    
    
    temp = struct( ...
            'type', 'ersp', ...
            'frequency', [1 30 33 60], ...
            'amplitude', amp, ...
            'modulation', 'none');
    
    tempc =  utl_create_component(1, temp, sourcefield);
    
    csignal = generate_signal_fromcomponent(tempc, e);
    csignal(1:time_on1*e.srate ) = 0;
    csignal(time_off1*e.srate-1:time_on2 ) = 0;
    csignal(time_off2*e.srate-1:end ) = 0;
    s2= struct();
    s2.data = csignal;
    s2.index = {'e', ':'};
    s2.amplitude = amp;
    s2.amplitudeType = 'relative';
    s2 = utl_check_class(s2, 'type', 'data');


end