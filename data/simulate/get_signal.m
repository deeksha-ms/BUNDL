function [sz_spikes, sz_sharps, sz_wave]  = get_signal(onset, szlen)
%fs = 200, totlen = 600 s, onset and szlen given in sec
% sharp wave, polyspikes, spikes+slowing (randomize number, order, amplitude)
max_spikes = 6*round(szlen/2);
min_spikes = 6*round(szlen/3);
offset = min(600, onset+szlen);
rng("shuffle")
sz_spikes = erp_get_class_random([min_spikes:max_spikes], ...  %number of spikes range
                                 [onset*1e03:offset*1e03], ... %seizure timepoints only
                                 [50:200], ...                 %width of spike range
		                         [-1:0.05:-0.7, 0.7:0.05:1]);  %amplitude range
rng("shuffle")
sz_sharps = erp_get_class_random([max_spikes:round(szlen)*6],...
                                  [onset*1e03:offset*1e03],...
                                  [200:1000], ...
		                         [-1:0.05:-0.7, 0.7:0.05:1]);

%%%%%%%%%change it to only within seizure duration
sz_wave = struct( ...
        'type', 'ersp', ...
        'frequency', [10 15 25 28], ...
        'amplitude', 1.0, ...
        'modulation', 'none');

end