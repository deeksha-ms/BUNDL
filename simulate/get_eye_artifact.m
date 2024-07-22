function eye_art  = get_eye_artifact(onset, len, noiselevel)

max_spikes = round(len)*10;
min_spikes = round(len/2)*10;
offset = min(600, onset+len);
if noiselevel == 1
    amp_range = [-1.1:0.05:-0.9, 0.9:0.05:1.1]; 
elseif noiselevel == 2
    amp_range = [-0.9:0.05:-0.75, 0.75:0.05:0.9];
elseif noiselevel ==3
    amp_range = [-0.75:0.05:-0.6, 0.6:0.05:0.75];
else 
    amp_range = [-0.6:0.05:-0.45, 0.45:0.05:0.6];
end


amp_range = [-1:0.1:-0.9, 0.9:0.1:1]; 

eye_art = erp_get_class_random([min_spikes:max_spikes], ...  %number of spikes range
                                 [onset*1e03:(offset)*1e03], ... %given timerange only
                                 [100:300], ...                 %width of spike range
		                         amp_range);  %amplitude range
end
