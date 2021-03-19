global title1;
global title2;
global title3;
global title4;

figure(2);
global dirname1;
cd(dirname1);

subplot(2,4,5);
plot_volts('X1_voltages.csv', "", false);


subplot(2,4,1);
plot_volts('X2_voltages.csv', title1, true);

set(gcf, 'Color', [1 1 1] .* 0.7);
set(gcf, 'InvertHardCopy', 'off');

global dirname2;
cd('~');
cd(dirname2);

subplot(2,4,6);
plot_volts('X1_voltages.csv', "", false);
    
subplot(2,4,2);
plot_volts('X2_voltages.csv', title2, true);

global dirname3;
cd('~');
cd(dirname3);

subplot(2,4,7);
plot_volts('X1_voltages.csv', "", false);

subplot(2,4,3);
plot_volts('X2_voltages.csv', title3, true);

set(gcf, 'Color', [1 1 1] .* 1.0);
set(gcf, 'InvertHardCopy', 'off');

global dirname4;
cd('~');
cd(dirname4);

subplot(2,4,8);
plot_volts('X1_voltages.csv', "", false);

subplot(2,4,4);
plot_volts('X2_voltages.csv', title4, true);

set(gcf, 'Color', [1 1 1] .* 1.0);
set(gcf, 'InvertHardCopy', 'off');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIGURE 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3); 
cd('~');
plot_receptive_field(dirname1);
set(gcf, 'Color', [1 1 1] .* 1.0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIGURE 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4);
cd('~');

subplot(2, 4, 5);
plot_histogram(dirname1, "A.", 2900, [1 0 0], 1);
subplot(2, 4, 1);
plot_histogram(dirname1, "A.", 2900, [0 0.6 0], 2);

subplot(2, 4, 6);
plot_histogram(dirname2, "B.", 2900, [1 0 0], 1);
subplot(2, 4, 2);
plot_histogram(dirname2, "B.", 2900, [0 0.6 0], 2);

subplot(2, 4, 7);
plot_histogram(dirname3, "C.", 1900, [1 0 0], 1);
subplot(2, 4, 3);
plot_histogram(dirname3, "C.", 1900, [0 0.6 0], 2);

subplot(2, 4, 8);
plot_histogram(dirname4, "D.", 2900, [1 0 0], 1);
subplot(2, 4, 4);
plot_histogram(dirname4, "D.", 2900, [0 0.6 0], 2);

set(gcf, 'Color', [1 1 1] .* 1.0);
% 
% figure(6);
% cd('~');
% subplot(2, 1, 1);
% plot_boxplot({dirname1, dirname2, dirname3, dirname4}, {'A', 'B', 'C', 'D'}, {2900, 2900, 1900, 2900}, [0 0 0], 1);
% 
% subplot(2, 1, 2);
% plot_boxplot({dirname1, dirname2, dirname3, dirname4}, {'A', 'B', 'C', 'D'}, {2900, 2900, 1900, 2900}, [0 0 0], 2);
% set(gcf, 'Color', [1 1 1] .* 1.0);

function blank = plot_volts(flname, ttl, top)
    y = read(flname);
    [m,n] = size(y);
    out = arrayfun(@(x) 1/(1 + exp(-(x - 20)/3)), y);
    out_avgs = mean(out, 1).';
    
    [mx, max_idx] = max(out_avgs(:,1));
    
    
    cull_inds = [];
%     y_firing_only = y;
%     %for i=1:n-1
%     %    if max(y(:, i)) < 20; cull_inds = [cull_inds i]; end
%     %end
%     y_firing_only(:, cull_inds) = [];
%     if plotarea
%         s = area(y(:, max_idx), 'FaceColor', [0.1 0.1 0.4]);
%         s.FaceAlpha = 0.7;
%     end
     
    n_colors = 10;
    cmap = hsv(n_colors);
    p = plot(y,...
        'LineWidth',2.0);
    for i=1:n
        p(i).Color = cmap(mod(i, n_colors - 1) + 1, :);
    end
    set(gca,'LineWidth',3);
    xaxis = [0 200 400 600 800 1000];
    xticks(xaxis);
    xticklabels(xaxis .* 0.03);
%    box on
    title(ttl, 'FontSize', 15, 'Units', 'normalized', 'Position', [0.5, 1.05, 0]);
    xlabel('Time (ms)', 'FontSize', 10);
    ylabel('Voltage (mV)', 'FontSize', 10);
    
    pos = get(gca, 'Position'); 
    if (top)
        pos(2) = 0.48;
    else 
        pos(2) = 0.1;
    end
    pos(4) = 0.3;
    set(gca, 'Position', pos)
end

function blank = plot_receptive_field(flname)
    W = read(flname + '/W1/weights_W1_e-2900.csv');
    [m, n] = size(W);
    W_arr = {};
    for i=1:m
        W_arr{i} = reshape(W(i, :).', [28 28]).';
    end
    for i=1:5
        for j = 1:5
            idx = (i - 1) * 5 + j;
            subplot(5, 5, idx);
            smoothing_filter = ones(1);
            J = filter2(smoothing_filter,W_arr{idx});
            imshow(J);
            colormap(gca, parula(1000));
        end
    end
end

 
function blank = plot_histogram(flname, ttl, epoch, col, idx)
    W = read(flname + '/W' + string(idx) + '/weights_W' + string(idx) + '_e-' + string(epoch) + '.csv');
    h = histogram(W(:), 'FaceColor', col, 'EdgeAlpha', 0);
    if(idx == 2)
        h.NumBins = 50;
    end
    title(ttl, 'FontSize', 15, 'Units', 'normalized', 'Position', [0.0, 1.05, 0]);
    xlabel('Weight', 'FontSize', 10);
    ylabel('Number of Synapses', 'FontSize', 10);
end

function blank = plot_boxplot(flnames, labels, epochs, col, idx)
    Ws = [];
    for i=1:numel(flnames)
        mat = read(flnames{i} + '/W' + string(idx) + '/weights_W' + string(idx) + '_e-' + string(epochs{i}) + '.csv');
        Ws = [Ws mat(:)]; % Add new column.
    end
    b = boxplot(Ws, 'Labels', labels,...
        'BoxStyle', 'outline', 'Widths', 0.3, 'Symbol', 'kx', 'Jitter', 0.2);
    if(idx == 1)
       title("First Layer Weights", 'FontSize', 15, 'Units', 'normalized', 'Position', [0.5, 1.05, 0]); 
    else
       title("Second Layer Weights", 'FontSize', 15, 'Units', 'normalized', 'Position', [0.5, 1.05, 0]); 
    end
    xlabel('Case', 'FontSize', 10);
    ylabel('Weight', 'FontSize', 10);
    lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
    set(lines, 'Color', 'r');
    set(lines, 'LineWidth', 3);
end

function out = read(name)
    out = readmatrix(name);
    out = out(:, 1:end-1); % Remove column of NaNs
end