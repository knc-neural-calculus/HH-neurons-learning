pwd
y_low_1 = readmatrix('GSL01_ID2108/X2_voltages.csv');
y_low_2 = readmatrix('GSL01_ID2108/X2_voltages.csv');
y_high_1 = readmatrix('GSL01_ID2108_GK_DOWN_10/X2_voltages.csv');
y_high_2 = readmatrix('GSL01_ID2108_GNA_DOWN_10/X2_voltages.csv');

idxs = max(y_high_1 > 20);

[UNUSED, n_hidden] = size(y_low_1);

figure(1);
subplot(3, 2, 3);
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

subplot(3, 2, 5);
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

subplot(3, 2, [4 6]);
y_high_2 = readmatrix('GSL01_ID2108_GNA_DOWN_20/X2_voltages.csv');
p2 = plot([y_low_2 y_high_2], 'LineWidth',1);
title({"Failure Example: Decreasing gNa by 20%"});
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

subplot(3, 2, [1 2]);
cd('varying');
lst = {'GNA', 'GK', 'GL', 'ENA', 'EK'};
hold on;
data = [];
cmap = hsv(5);
for i=1:5
    percents = [];
    X_i = [];
    for j=0:80
        mat = readmatrix(strcat(lst{i}, '/',...
        lst{i}, '_VARIED_ID', int2str(j), '/percent2900.txt'));
        percents = [percents; mat(end, 2)];
        X_i = [X_i; -10 + 0.5 * j];
    end
    plot(linspace(-20, 20, 81), percents, '--o', 'Color', 0.5 .* cmap(i, :), 'MarkerFaceColor', 0.5 .* cmap(i, :), 'MarkerEdgeColor', [1 1 1]);
    axis([-20 20 44.5 70])
    title("Varying Physiology Parameters")
    xlabel("Percent Variation")
    ylabel("Accuracy")
    %    sct = scatter(linspace(-20, 20, 81), percents,);
end
legend({'gNa', 'gK', 'gL', 'ENa', 'EK'})
