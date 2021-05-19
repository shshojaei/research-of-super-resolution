function [features] = extract(conf, X, scale, filters)

% Compute one grid for all filters
grid = sampling_grid(size(X), ...
    conf.window, conf.overlap, conf.border, scale);

% Current image features extraction [feature x index]
if isempty(filters)
    feature_size = prod(conf.window);
    f = X(grid);
    features = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
else
    feature_size = prod(conf.window) * numel(conf.filters);
    features = zeros([feature_size size(grid, 3)], 'single');
    for i = 1:numel(filters)
        f = conv2(X, filters{i}, 'same');
        f = f(grid);
        f = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
        features((1:size(f, 1)) + (i-1)*size(f, 1), :) = f;
    end
end
