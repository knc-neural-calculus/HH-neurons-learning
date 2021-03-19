global dirname1;
global title1;
global title2;
cd(dirname1);

subplot(2,2,4);
plot_volts('X1_voltages.csv');
xlabel('Time (ms)', 'FontSize', 24);
    
subplot(2,2,2);
title(title2, 'FontSize', 30);
plot_volts('X2_voltages.csv');

global dirname2;
cd('~');
cd(dirname2);

subplot(2,2,3);
plot_volts('X1_voltages.csv');
xlabel('Time (ms)', 'FontSize', 24);
ylabel('Voltage (mV)', 'FontSize', 24);

subplot(2,2,1);
title(title1, 'FontSize', 30);
plot_volts('X2_voltages.csv');
ylabel('Voltage (mV)', 'FontSize', 24);

set(gcf, 'Color', [1 1 1] .* 1.0);
set(gcf, 'InvertHardCopy', 'off');

% figure(2);
% imshow(out_avgs ./ max(out_avgs),'InitialMagnification',1000);
% cmap = colormap('gray');
% [m_cmap, n_cmap] = size(cmap);
% cmap(m_cmap, :) = [1 0 0];
% colormap(cmap);
% colormap(gca);
function blank = plot_volts(flname)
    y = readmatrix(flname);
    [m,n] = size(y);
    y = y(:, 1:n - 1); % Remove last column (an extra column of NaNs is added).
    out = arrayfun(@(x) 1/(1 + exp(-(x - 20)/3)), y);
    out_avgs = mean(out, 1).';
    
    [mx, max_idx] = max(out_avgs(:,1));
    
    
    hold on
    cull_inds = [];
    y_firing_only = y;
    %for i=1:n-1
    %    if max(y(:, i)) < 20; cull_inds = [cull_inds i]; end
    %end
    y_firing_only(:, cull_inds) = [];
%    s = area(y(:, max_idx), 'FaceColor', [0.1 0.1 0.4]);
    s.FaceAlpha = 0.7;
    
    p = plot(y_firing_only,...
        'LineWidth',3);
    set(gca, 'Color', [1 1 1] .* 0.9);
    [cull_m, cull_n] = size(y_firing_only);
    for i=1:cull_n
        p(i).Color = [1 0 0] + ([0 0 0] - [1 0 0]) .* i / cull_n;
    end
    set(gca,'LineWidth',3);
    xaxis = [0 200 400 600 800 1000];
    xticks(xaxis);
    xticklabels(xaxis .* 0.03);
    box on
       
    hold off
end