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