cd('SAN/SAN');
figure('units','normalized','outerposition',[0 0 1 1]);

subplot(3, 6, [2 3]);
san_sample = readmatrix('san_sample.csv');
san_sample_1 = san_sample(20000:40000, :);
plot(san_sample_1(:, 1), san_sample_1(:, 2), 'Color', 'Black', 'MarkerSize', 10);

subplot(3, 6, [4 5]);
san_sample_2 = san_sample(100000:200000, :);
plot(san_sample_2(:, 1), san_sample_2(:, 2), 'Color', 'Black', 'MarkerSize', 10);
th = title('SAN Model Average Bursting Dynamics', 'FontSize', 30);
titlePos = get(th, 'position');
titlePos(1) = 930;
set( th , 'position' , titlePos);

subplot(3, 6, [7 8]);
san_iv = readmatrix('san_iv_plot_no_zoom.csv');
plot(san_iv(:, 1), san_iv(:, 2), '.', 'Color', 'Black', 'MarkerSize', 3);
xlabel('Input Current (μA/cm^2)');
ylabel('Average Output');

subplot(3, 6, [9 10]);
san_iv = readmatrix('san_iv_plot_zoom_1.csv');
plot(san_iv(:, 1), san_iv(:, 2), '.', 'Color', 'Black', 'MarkerSize', 1);
th = title('SAN Model Average Output Versus Current', 'FontSize', 30);
titlePos = get( th , 'position');
titlePos(2) = titlePos(2) + 0.005;
%set( th , 'position' , titlePos);
xlabel('Input Current (μA/cm^2)');
ylabel('Average Output');

subplot(3, 6, [11 12]);
san_iv = readmatrix('san_iv_plot_zoom_2.csv');
plot(san_iv(:, 1), san_iv(:, 2), '.', 'Color', 'Black', 'MarkerSize', 1);
xlabel('Input Current (μA/cm^2)');
ylabel('Average Output');

subplot(3, 6, [13 14 15 16]);
hh_iv = readmatrix('hh_iv_plot.csv');
plot(hh_iv(:, 1), hh_iv(:, 2), '.', 'Color', 'Black', 'MarkerSize', 5);
th = title('HH Model Average Output Versus Current', 'FontSize', 15);
xlabel('Input Current (μA/cm^2)');
ylabel('Average Output');

% These were the max accuracies achieved. The former was achieved using
% nevergrad with the SAN model. The latter resulted from the inihibition
% parameter sweep data. 
accuracies = [18 72];
subplot(3, 6, [17 18]);
bar(accuracies, 'FaceColor', [0.8 0 0.2]);
ylabel('Accuracy (%)');
xlabel('Model Used');
title('Max Accuracy: SAN vs HH', 'FontSize', 15);
