function seiz_sig = generate_seizure(szon,szoff, epochs)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
len= szoff - szon; 
%mid = szon+round(len/2);
minspikes = round(3.5*len); 
maxspikes = 4  * len; 

s1 =  erp_get_class_random([minspikes:maxspikes], ...  %number of spikes range
                                 [szon*1e03:szoff*1e03], ... %seizure timepoints only
                                 [20:70], ...                 %width of spike range
		                         [-1.:0.05:-0.9, 0.9:0.05:1.]);  %amplitude range
seizure = generate_signal_fromclass(s1, epochs); 

minspikes = round(2.5*len); 
maxspikes = 3 * len; 
s2 =  erp_get_class_random([minspikes:maxspikes], ...  %number of spikes range
                                 [szon*1e03:szoff*1e03], ... %seizure timepoints only
                                 [70:200], ...                 %width of spike range
		                         [-0.95:0.05:-0.75, 0.75:0.05:0.95]);  %amplitude range
seizure = seizure + generate_signal_fromclass(s2, epochs); 

nbursts = randi([1, 5]); 
for j=1:nbursts
    ersp = struct( ...
            'type', 'ersp', ...
            'frequency', [3 3.5 4 5], ...
            'amplitude', 1.0, ...
            'modulation', 'none');
    ersp.modulation = 'burst';
    ersp.modLatency = randi([szon*1e03, szoff*1e03]);      % centre of the burst, in ms
    ersp.modWidth = 200;        % width (half duration) of the burst, in ms
    ersp.modTaper = 0.2;        % taper of the burst
    
    ersp = utl_check_class(ersp);

    seizure = seizure + generate_signal_fromclass(ersp, epochs);

end
seizure = seizure/max(seizure); 

seiz_sig = struct();
seiz_sig.data = seizure;
seiz_sig.index = {'e', ':'};
seiz_sig.amplitude = 1.0;
seiz_sig.amplitudeType = 'relative';
seiz_sig = utl_check_class(seiz_sig, 'type', 'data');

end