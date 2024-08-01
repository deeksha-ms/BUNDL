function [musc_art , s2] = get_musc_artifact(onset, len, sourcefield, e, noiselevel)
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

max_spikes = round(len)*50;

offset = min(600, onset+len);
musc_art = erp_get_class_random([max_spikes-10:max_spikes], ...  %number of spikes range
                                 [onset*1e03:(offset)*1e03], ... %given timerange only
                                 [20:50], ...                 %width of spike range
		                         amp_range);  %amplitude range



temp = struct( ...
            'type', 'ersp', ...
            'frequency', [1 30 33 60], ...
            'amplitude', amp, ...
            'modulation', 'none');
    
tempc =  utl_create_component(1, temp, sourcefield);

csignal = generate_signal_fromcomponent(tempc, e);
csignal(1:onset*e.srate ) = 0;
csignal(offset*e.srate-1:end ) = 0;
s2= struct();
s2.data = csignal;
s2.index = {'e', ':'};
s2.amplitude = amp;
s2.amplitudeType = 'relative';
s2 = utl_check_class(s2, 'type', 'data');

end
