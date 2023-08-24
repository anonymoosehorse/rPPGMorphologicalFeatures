files = dir("/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/vicar/split_traces5_1D_gt_csv/");
%files = dir("C:/Users/ruben/Documents/thesis/data/vipl/split_traces/");
display(files)
for i = 1:length(files)
    if endsWith(files(i).("name"), ".csv")
        filepath = files(i).("folder") + "/" + files(i).("name");
        x = readtable(filepath);
        x = table2array(x);
        my_cwt = signal_to_cwt(x(1,:)/1000, x(2,:));
        
        times = string(round(0:my_cwt.dt:my_cwt.dt*255, 1));
        freqs = round(my_cwt.frequencies, 1);
        XLabels = 1:length(times);
        YLabels = 1:length(freqs);
        times(mod(XLabels,20) ~= 0) = " ";        
        freqs(mod(YLabels,20) ~= 0) = " ";
        %figure()
        %h1 = heatmap(real(my_cwt.cfs));
        %h1.GridVisible = 'off';
        %h1.XDisplayLabels = times;
        %h1.YDisplayLabels = freqs;
        %xticks([0, 2, 4, 6, 8, 10])
        %xticklabels({''})
        %xlabel('Time (s)')
        %ylabel('Frequency (Hz)')
        
        writematrix(real(my_cwt.cfs), "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/vicar/split_cwt5_gt/r_" + files(i).("name"))
        writematrix(imag(my_cwt.cfs), "/tudelft.net/staff-umbrella/StudentsCVlab/rsangers/data/vicar/split_cwt5_gt/i_" + files(i).("name"))
    end
end

%x2 = icwt(my_cwt.cfs);
%figure()
%plot(x2)
%figure()
%plot(x(2, :))
%h1 = heatmap(real(my_cwt.cfs));
%h1.GridVisible = 'off'
display("done")
