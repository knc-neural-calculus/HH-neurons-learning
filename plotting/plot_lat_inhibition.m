y_low_1 = read('GSL01_ID2108/X2_voltages.csv');
y_low_2 = read('GSL01_ID2108/X2_voltages.csv');
y_high_1 = read('GSL01_ID2108_GK_DOWN_10/X2_voltages.csv');
y_high_2 = read('GSL01_ID2108_GNA_DOWN_10/X2_voltages.csv');

idxs = max(y_high_1 > 20);

[UNUSED, n_hidden] = size(y_low_1);


figure(1);
subplot(2, 1, 1);
p1 = plot([y_low_1 y_high_1], 'LineWidth',1);
title({"Decreasing gK by 10%"});
for i=1:10
    p1(i).Color = [1 0 0];
    p1(10 + i).Color = [0 0 1];
end
legend([p1(1) p1(11)], "No Modification", "Modified");
xaxis = [0 200 400 600 800 1000];
xticks(xaxis);
xticklabels(xaxis .* 0.03);
xlabel('Time (ms)', 'FontSize', 10);
ylabel('Voltage (mV)', 'FontSize', 10);

subplot(2, 1, 2);
p2 = plot([y_low_2 y_high_2], 'LineWidth',1);
title({"Decreasing gNa by 10%"});
for i=1:10
    p2(i).Color = [1 0 0];
    p2(10 + i).Color = [0 0 1];
end
legend([p2(1) p2(11)], "No Modification", "Modified");
xaxis = [0 200 400 600 800 1000];
xticks(xaxis);
xticklabels(xaxis .* 0.03);
xlabel('Time (ms)', 'FontSize', 10);
ylabel('Voltage (mV)', 'FontSize', 10);

% figure(2);
% W_low_2 = read('GSL01_ID1593/W2/weights_W2_e-2900.csv');
% is = imshow(W_low_2)
% 
% figure(3)
% imshow(is.CData(:, idxs))
% 
% figure(4) 
% W_low_1 = read('GSL01_ID1593/W1/weights_W1_e-2900.csv');
% W_high_1 = read('CUR_5_ID1/W1/weights_W1_e-2900.csv');
% 
% boxplot([W_low_1(:) W_high_1(:)], 'Labels', {'Low Output Inhibition', 'High Output Inhibition'},...
%         'BoxStyle', 'outline', 'Widths', 0.3, 'Symbol', 'kx', 'Jitter', 0.2);
% title('Input Layer Weights Box Plots for Varied Output Inhibition')
% 
function out = read(name)
    out = readmatrix(name);
    out = out(:, 1:end-1); % Remove column of NaNs
end