clear; clc; close all;

disp('Select the input nuclear image.');
[infile, inpath] = uigetfile( ...
    {'*.tif;*.tiff', 'TIFF files (*.tif, *.tiff)'}, ...
    'Select nuclear image');

if isequal(infile, 0)
    error('No input image was selected. Program terminated.');
end
input_tif = fullfile(inpath, infile);

disp('Select the output location for the corner-coordinate file.');
[outfile, outpath] = uiputfile( ...
    {'*.tsv', 'TSV files (*.tsv)'}, ...
    'Save corner_points.tsv', ...
    fullfile(inpath, 'corner_points.tsv'));

if isequal(outfile, 0)
    error('No output file was selected. Program terminated.');
end
output_tsv = fullfile(outpath, outfile);

fprintf('Reading image: %s\n', input_tif);
info = imfinfo(input_tif);
fprintf('Image dimensions (Width x Height): %d x %d\n', ...
    info.Width, info.Height);

I = imread(input_tif);

if ndims(I) == 3
    I = I(:, :, 1);
end

I = squeeze(I);

if ndims(I) ~= 2
    error('The input image is not a two-dimensional grayscale image.');
end

[h, w] = size(I);
fprintf('MATLAB image dimensions (Height x Width): %d x %d\n', h, w);

% Estimate the display intensity range using sparse sampling.
sy = max(1, round(h / 1200));
sx = max(1, round(w / 1200));

sample = double(I(1:sy:end, 1:sx:end));
sample = sort(sample(:));
n = numel(sample);

lo = sample(max(1, round(0.01 * n)));
hi = sample(min(n, round(0.995 * n)));

if hi <= lo
    lo = double(min(I(:)));
    hi = double(max(I(:)));
end

% Generate a downsampled image for whole-image visualization.
ds = max(1, ceil(max(h, w) / 2500));
I_over = I(1:ds:end, 1:ds:end);

% Define initial search windows for the four corner spots.
xw = max(2200, round(0.07 * w));
yw = max(1400, round(0.16 * h));

xw = min(xw, w);
yw = min(yw, h);

labels = {'TL', 'TR', 'BL', 'BR'};

descs = { ...
    'Top-left corner spot center', ...
    'Top-right corner spot center', ...
    'Bottom-left corner spot center', ...
    'Bottom-right corner spot center'};

windows = [ ...
    1,        xw,       1,        yw; ...
    w-xw+1,   w,        1,        yw; ...
    1,        xw,       h-yw+1,   h; ...
    w-xw+1,   w,        h-yw+1,   h];

figure(10); clf;

imshow(I_over, [lo hi], ...
    'InitialMagnification', 'fit', ...
    'XData', [1 w], ...
    'YData', [1 h]);

axis on;
axis image ij;

xlabel('x (pixels)');
ylabel('y (pixels)');
title('Whole-image overview (red boxes indicate initial corner search regions)');

hold on;

for k = 1:4
    x1 = windows(k,1);
    x2 = windows(k,2);
    y1 = windows(k,3);
    y2 = windows(k,4);

    rectangle( ...
        'Position', [x1, y1, x2-x1, y2-y1], ...
        'EdgeColor', 'r', ...
        'LineWidth', 1.5);

    text( ...
        x1 + 50, ...
        y1 + 50, ...
        labels{k}, ...
        'Color', 'y', ...
        'FontWeight', 'bold', ...
        'FontSize', 12);
end

disp('---------------------------------------------');
disp('Instructions:');
disp('1) The four corners are processed sequentially: TL, TR, BL, and BR.');
disp('2) An initial local window is displayed for each corner.');
disp('3) Enter z to draw a rectangular region and zoom in.');
disp('4) The zoom operation can be repeated multiple times.');
disp('5) Enter c when ready, then click once at the center of the corner spot.');
disp('6) Press Enter to confirm the selected point, or enter n to select again.');
disp('7) Enter r to reset the current view to the initial corner window.');
disp('---------------------------------------------');

pts = zeros(4,2);

for k = 1:4

    base_view = windows(k,:);
    view = base_view;
    point_ok = false;

    while ~point_ok

        x1 = view(1);
        x2 = view(2);
        y1 = view(3);
        y2 = view(4);

        crop = I(y1:y2, x1:x2);

        figure(1); clf;

        imshow(crop, [lo hi], ...
            'InitialMagnification', 'fit', ...
            'XData', [x1 x2], ...
            'YData', [y1 y2]);

        axis on;
        axis image ij;

        xlabel('x (pixels)');
        ylabel('y (pixels)');

        title({ ...
            sprintf('%s: %s', labels{k}, descs{k}); ...
            'Commands: z=zoom, r=reset, c=select point, q=quit'}, ...
            'Interpreter', 'none');

        fprintf('\n==============================\n');
        fprintf('Current corner: %s: %s\n', labels{k}, descs{k});
        fprintf('Current view: x=[%d, %d], y=[%d, %d]\n', ...
            x1, x2, y1, y2);

        cmd = input('Enter z / r / c / q: ', 's');

        if strcmpi(cmd, 'q')

            error('Program terminated by the user.');

        elseif strcmpi(cmd, 'r')

            view = base_view;
            continue;

        elseif strcmpi(cmd, 'z')

            disp('Draw a rectangle in Figure 1 around the region to enlarge.');

            rect = getrect(gca);

            if isempty(rect) || rect(3) < 5 || rect(4) < 5
                disp('The selected region is too small or invalid. Keeping the current view.');
                continue;
            end

            new_x1 = max(1, floor(rect(1)));
            new_y1 = max(1, floor(rect(2)));
            new_x2 = min(w, ceil(rect(1) + rect(3)));
            new_y2 = min(h, ceil(rect(2) + rect(4)));

            if new_x2 <= new_x1 || new_y2 <= new_y1
                disp('The new view is invalid. Keeping the current view.');
                continue;
            end

            view = [new_x1, new_x2, new_y1, new_y2];
            continue;

        elseif strcmpi(cmd, 'c')

            disp('Click once in Figure 1 at the geometric center of the corner spot.');

            [x, y, button] = ginput(1);

            if isempty(x) || isempty(y) || isempty(button)
                disp('No valid point was selected. Returning to the current view.');
                continue;
            end

            hold on;
            plot(x, y, 'r+', 'MarkerSize', 16, 'LineWidth', 2);
            drawnow;

            fprintf('Selected point: %s, px=%d, py=%d\n', ...
                labels{k}, round(x), round(y));

            cmd2 = input( ...
                'Press Enter to confirm; n=select again; z=continue zooming; r=reset; q=quit: ', ...
                's');

            if strcmpi(cmd2, 'q')

                error('Program terminated by the user.');

            elseif strcmpi(cmd2, 'n')

                continue;

            elseif strcmpi(cmd2, 'z')

                continue;

            elseif strcmpi(cmd2, 'r')

                view = base_view;
                continue;

            else

                pts(k,:) = round([x, y]);
                point_ok = true;

            end

        else

            disp('Invalid command. Enter z / r / c / q.');
            continue;

        end
    end
end

fid = fopen(output_tsv, 'w');

if fid == -1
    error('Unable to write output file: %s', output_tsv);
end

fprintf(fid, 'label\tpx\tpy\n');

for k = 1:4
    fprintf(fid, '%s\t%d\t%d\n', ...
        labels{k}, pts(k,1), pts(k,2));
end

fclose(fid);

disp(' ');
disp('Corner-coordinate file saved to:');
disp(output_tsv);

T = table( ...
    labels', ...
    pts(:,1), ...
    pts(:,2), ...
    'VariableNames', {'label', 'px', 'py'});

disp(T);

figure(11); clf;

imshow(I_over, [lo hi], ...
    'InitialMagnification', 'fit', ...
    'XData', [1 w], ...
    'YData', [1 h]);

axis on;
axis image ij;

xlabel('x (pixels)');
ylabel('y (pixels)');
title('Selected corner spots in the whole image');

hold on;

plot( ...
    pts(:,1), ...
    pts(:,2), ...
    'ro', ...
    'MarkerSize', 8, ...
    'LineWidth', 1.5);

text( ...
    pts(:,1) + 50, ...
    pts(:,2), ...
    labels, ...
    'Color', 'y', ...
    'FontSize', 12, ...
    'FontWeight', 'bold');
