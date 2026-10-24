%% DMD_VIDEO.M  --  Dynamic Mode Decomposition on video (+ Robust PCA comparison)
% Needs: base MATLAB + Image Processing Toolbox (imresize, imshow only).
% If 'input.mp4' is missing, a synthetic demo video is generated instead
% (static gradient + 3 Hz flickering patch + 0.5 Hz moving blob).
clear; clc; close all;

%% ---------------- User parameters ----------------
videoFile    = 'input.mp4';   % <-- your video path
scale        = 0.15;     % spatial downsample factor (memory!)
maxFrames    = 300;      % max number of frames used (10 s at 30 fps)
frameSkip    = 1;        % use every n-th frame (effective fps = fps/frameSkip)
rankR        = 20;       % DMD rank. 0 = automatic (usually too high with noise). Try 15-30;
                         % too high fits noise (random fast/decaying modes), too low loses motion.
energyThresh = 0.99999;  % cumulative energy used for auto rank (background dominates, so keep this high)
rankMax      = 30;       % cap on auto rank (only used if rankR = 0)
subtractMean = true;     % remove a static image before DMD (added back to the background)
bgStat       = 'median'; % static background: 'quantile', 'median' or 'mean'
bgQuantile   = 0.15;     % for 'quantile': 0.15 removes BRIGHT movers on a darker scene;
                         % use 0.85 for DARK movers on a lighter scene
useFB        = true;     % forward-backward DMD: removes the noise-induced damping bias
delays       = 4;        % time-delay (Hankel) depth; 1 = plain DMD. Pendulum: 8, turbine: 4
bgFreqTol    = 0.05;     % Hz   : |f| below this -> background candidate
bgGrowthTol  = 0.05;     % 1/s  : |Re(omega)| below this -> background candidate (true static = 0)
useBgModes   = false;    % false = background is ONLY the static image (no DMD modes added)
bandHz       = [0.3 4.0];% Hz: band(s) to keep. One row [f1 f2] or several rows, e.g. [0.5 0.7; 1.1 1.3]
doRPCA       = true;     % run Robust PCA comparison
writeOutput  = true;     % write dmd_result.mp4
colorOutput  = true;     % also show/write COLOUR results (analysis itself stays grayscale)
doTracking   = false;    % single-object swing tracker (section 7) - for the pendulum only
playVideo    = true;     % play the WHOLE result as a movie in a MATLAB window
videoUpscale = 4;        % enlarge the saved videos (pixel replication) so they are easy to watch
hiResVideo   = true;     % also write dmd_result_hq.mp4: SHARP full-colour video (analysis stays small)
displayScale = 0.5;      % size of that sharp video relative to the original (0.5 = half size)

%% ---------------- 1. Load / downsample video ----------------
if isfile(videoFile)
    vr  = VideoReader(videoFile);
    fps = vr.FrameRate / frameSkip;
    V = []; k = 0; idx = 0;
    while hasFrame(vr) && k < maxFrames
        f = readFrame(vr); idx = idx + 1;
        if mod(idx-1, frameSkip) ~= 0, continue; end
        if size(f,3) == 3, fgray = rgb2gray(f); else, fgray = f; end
        g = imresize(im2double(fgray), scale);
        k = k + 1;
        if k == 1
            [h, w] = size(g);
            V = zeros(h*w, maxFrames);
            Vrgb = zeros(h, w, 3, maxFrames, 'uint8');   % colour copy, for display only
        end
        V(:,k) = g(:);
        cf = imresize(f, [h w]);
        if isa(cf, 'uint16'), cf = uint8(double(cf)/257);
        elseif ~isa(cf, 'uint8'), cf = uint8(255*min(max(double(cf),0),1)); end
        if size(cf,3) == 1, cf = repmat(cf, [1 1 3]); end
        Vrgb(:,:,:,k) = cf;
    end
    V = V(:,1:k);
    Vrgb = Vrgb(:,:,:,1:k);
else
    warning('"%s" not found. Using synthetic demo video.', videoFile);
    fps = 30; h = 60; w = 80; k = maxFrames;
    t = (0:k-1)/fps;
    [xx, yy] = meshgrid(1:w, 1:h);
    bg = 0.3 + 0.3*xx/w;                               % static gradient
    patch = double(xx>50 & xx<65 & yy>10 & yy<25);    % flicker region
    V = zeros(h*w, k);
    for j = 1:k
        blobX = w/2 + 25*sin(2*pi*0.5*t(j));
        blob  = exp(-((xx-blobX).^2 + (yy-45).^2)/(2*4^2));
        fr = bg + 0.25*patch*sin(2*pi*3*t(j)) + 0.4*blob;
        V(:,j) = fr(:);
    end
    Vrgb = repmat(reshape(uint8(255*min(max(V,0),1)), h, w, 1, k), [1 1 3 1]);
end
[n, K] = size(V);
dt = 1/fps;
tvec = (0:K-1)*dt;
fprintf('Frame size %dx%d, %d frames, fps = %.2f, Nyquist = %.2f Hz\n', ...
        h, w, K, fps, fps/2);

%% ---------------- 2. Exact DMD (mean removal + time-delay embedding) ----------------
if subtractMean
    switch bgStat
        case 'quantile'
            Vs = sort(V, 2);
            Vm = Vs(:, min(K, max(1, round(bgQuantile*K))));
            clear Vs
        case 'median'
            Vm = median(V, 2);
        otherwise
            Vm = mean(V, 2);
    end
else
    Vm = zeros(n,1);
end
Vc = V - Vm;
if delays > 1
    Kh = K - delays + 1;
    H = zeros(n*delays, Kh);
    for i = 1:delays
        H((i-1)*n+1 : i*n, :) = Vc(:, i:i+Kh-1);   % stack shifted copies of the video
    end
else
    H = Vc;  Kh = K;
end
clear Vc
tH = (0:Kh-1)*dt;

X1 = H(:,1:end-1);
X2 = H(:,2:end);

[U, S, W] = svd(X1, 'econ');
s = diag(S);
if rankR == 0
    rankR = find(cumsum(s.^2)/sum(s.^2) >= energyThresh, 1);
    if isempty(rankR), rankR = numel(s); end
    rankR = min(rankR, rankMax);
end
r = min(rankR, numel(s));
fprintf('DMD rank r = %d  (delays = %d, mean removed = %d, fwd-bwd = %d)\n', r, delays, subtractMean, useFB);

figure('Name','Singular values');
semilogy(s, 'o-'); hold on; grid on; xline(r, 'r--', sprintf('r = %d', r));
xlabel('index'); ylabel('singular value'); title('SVD spectrum (choose r near the knee)');

Ur = U(:,1:r);  Sr = S(1:r,1:r);  Wr = W(:,1:r);
Atilde = Ur' * X2 * Wr / Sr;          % r x r reduced operator
if useFB
    % Forward-backward DMD: A_fb = sqrtm( A_forward * inv(A_backward) )
    Abwd   = (Ur' * X1) / (Ur' * X2);
    Atilde = sqrtm(Atilde / Abwd);
end
[Wev, Dg] = eig(Atilde);
lambda = diag(Dg);

Phi   = X2 * Wr / Sr * Wev;           % exact DMD modes (n*delays x r)
omega = log(lambda) / dt;             % continuous-time eigenvalues sigma + i*omega
freq  = imag(omega) / (2*pi);         % Hz  (aliased if above Nyquist)
sigma = real(omega);                  % growth rate (1/s)

% Mode amplitudes: least-squares fit over ALL snapshots (robust; fitting only the
% first frame fails if it is black / a fade-in, giving b = 0 and 100% error)
VandH = exp(omega * tH);                        % r x Kh
Pm    = (Phi' * Phi) .* conj(VandH * VandH');
qv    = diag(Phi' * H * VandH');
b     = Pm \ qv;
phiNorm = sqrt(sum(abs(Phi).^2,1))';
clear H X1 X2 U W
Phi   = Phi(1:n,:);                   % keep the first block = the spatial image

% Time dynamics on the full time axis: x(t) = sum_k b_k phi_k exp(omega_k t)
Vand  = exp(omega * tvec);                      % r x K
Tdyn  = b .* Vand;
power = (abs(b) .* phiNorm).^2;

% Diagnostic: strongest modes with f >= 0 (use this to pick bandHz and check rank)
cand = find(freq >= -1e-9);
[~, oP] = sort(power(cand), 'descend');
fprintf('\nTop modes by power (f >= 0):\n   f (Hz)    growth(1/s)   |lambda|\n');
for q = 1:min(12, numel(oP))
    m = cand(oP(q));
    fprintf('%9.3f %12.3f %10.4f\n', freq(m), sigma(m), abs(lambda(m)));
end
fprintf('\n');

%% ---------------- 3. Background / foreground / band ----------------
bgIdx = abs(freq) < bgFreqTol & abs(sigma) < bgGrowthTol & useBgModes;
if ~any(bgIdx) && ~subtractMean
    [~, im] = min(abs(omega));  bgIdx = false(r,1);  bgIdx(im) = true;
end
bandIdx = false(r,1);
for q = 1:size(bandHz,1)
    bandIdx = bandIdx | (abs(freq) >= bandHz(q,1) & abs(freq) <= bandHz(q,2));
end

Xbg   = Vm + real(Phi(:,bgIdx)   * Tdyn(bgIdx,:));   % mean image + 0 Hz modes
Xband = real(Phi(:,bandIdx) * Tdyn(bandIdx,:));
Xfg   = V - Xbg;                                      % foreground = residual
Xfull = Vm + real(Phi * Tdyn);
fprintf('Background modes: %d, band modes: %d\n', nnz(bgIdx), nnz(bandIdx));
fprintf('Full DMD reconstruction error: %.3f %%\n', ...
        100*norm(V - Xfull,'fro')/norm(V,'fro'));
fprintf('Error relative to the moving part only (V - static image): %.1f %%\n', ...
        100*norm(V - Xfull,'fro')/norm(V - Vm,'fro'));

%% ---------------- 4. Robust PCA comparison ----------------
if doRPCA
    tic;
    [L, Ssp] = rpca_ialm(V, 1/sqrt(max(n,K)), 1e-6, 200);
    fprintf('RPCA time: %.1f s\n', toc);
end

%% ---------------- 5. Plots ----------------
tile = @(x) reshape(x, h, w);

% (a) Eigenvalues on the unit circle
figure('Name','DMD eigenvalues');
th = linspace(0, 2*pi, 400);
h0 = plot(cos(th), sin(th), '--', 'Color', [.8 .8 .8]); hold on; axis equal; grid on;
h1 = plot(real(lambda), imag(lambda), 'bo', 'MarkerFaceColor', [.6 .6 1]);
hs = [h0 h1]; names = {'unit circle','all modes'};
if any(bgIdx)
    hs(end+1) = plot(real(lambda(bgIdx)), imag(lambda(bgIdx)), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
    names{end+1} = 'background';
end
if any(bandIdx)
    hs(end+1) = plot(real(lambda(bandIdx)), imag(lambda(bandIdx)), 'g^', 'MarkerSize', 10, 'LineWidth', 2);
    names{end+1} = 'selected band';
end
xlabel('Re(\lambda)'); ylabel('Im(\lambda)');
title('DMD eigenvalues and the unit circle');
legend(hs, names, 'Location','best');
axis normal; xlim([0.85 1.05]); ylim([-0.4 0.4]);   % zoom near lambda = 1 (modes cluster there at 30 fps)
title('DMD eigenvalues, zoomed near \lambda = 1');

% (b) Power vs frequency
figure('Name','Power vs frequency');
pp = max(power, eps);
semilogy(freq, pp, 'bo', 'MarkerFaceColor', 'b'); hold on; grid on;
for q = 1:r, semilogy([freq(q) freq(q)], [min(pp)/10 pp(q)], 'b-'); end
xline(fps/2, 'r--', 'Nyquist'); xline(-fps/2, 'r--');
xlim([-fps/2*1.1, fps/2*1.1]);
xlabel('Frequency (Hz)'); ylabel('Mode power |b_k|^2 ||\phi_k||^2');
title('DMD power spectrum');

% (c) Spatial mode images (top 6 positive-frequency modes by power)
figure('Name','Spatial modes');
cand = find(freq >= -1e-9);
[~, ord] = sort(power(cand), 'descend');
show = cand(ord(1:min(6, numel(ord))));
for q = 1:numel(show)
    subplot(2, 3, q);
    imagesc(tile(abs(Phi(:,show(q))))); axis image off; colormap(gca, 'hot');
    title(sprintf('f = %.2f Hz, |\\lambda| = %.3f', freq(show(q)), abs(lambda(show(q)))));
end
sgtitle('Spatial DMD modes |\phi_k|');

% (d) Reconstructions at a mid frame
j = round(K/2);
fgScale = max(abs(Xfg(:))) + eps;
bnScale = max(abs(Xband(:))) + eps;
figure('Name','Reconstructions');
vmax = max(V(:));   % same grey scale for all natural-image panels
subplot(2,3,1); imshow(tile(V(:,j)),[0 vmax]);              title('Original');
subplot(2,3,2); imshow(tile(Xbg(:,j)),[0 vmax]);            title('DMD background');
subplot(2,3,3); imshow(tile(abs(Xfg(:,j))),[0 fgScale]);   title('DMD foreground');
subplot(2,3,4); imshow(tile(abs(Xband(:,j))),[0 bnScale]); title(sprintf('Band %.1f-%.1f Hz', min(bandHz(:,1)), max(bandHz(:,2))));
if doRPCA
    subplot(2,3,5); imshow(tile(L(:,j)),[0 vmax]);          title('RPCA low-rank');
    subplot(2,3,6); imshow(tile(abs(Ssp(:,j))),[0 fgScale]); title('RPCA sparse');
end

%% ---------------- 5b. Colour display (analysis above stays grayscale) ----------------
if colorOutput
    bgRGB = zeros(h, w, 3);                       % static colour background (same statistic as bgStat)
    for c = 1:3
        Mc = reshape(double(Vrgb(:,:,c,:))/255, h*w, K);
        bgRGB(:,:,c) = reshape(staticImage(Mc, bgStat, bgQuantile), h, w);
    end
    heat = hot(256);
    rgbj = double(Vrgb(:,:,:,j))/255;
    [fgC, bandC] = colorPanels(rgbj, bgRGB, Xfg(:,j), Xband(:,j), fgScale, bnScale, h, w, heat);
    figure('Name','Reconstructions (colour)');
    subplot(2,3,1); imshow(rgbj);  title('Original');
    subplot(2,3,2); imshow(bgRGB); title('DMD background');
    subplot(2,3,3); imshow(fgC);   title('DMD foreground');
    subplot(2,3,4); imshow(bandC);
    title(sprintf('Band %.1f-%.1f Hz', min(bandHz(:,1)), max(bandHz(:,2))));
    if doRPCA
        [spC, ~] = colorPanels(rgbj, bgRGB, Ssp(:,j), Xband(:,j), fgScale, bnScale, h, w, heat);
        subplot(2,3,5); imshow(tile(L(:,j)),[0 vmax]); title('RPCA low-rank');
        subplot(2,3,6); imshow(spC);                    title('RPCA sparse');
    end
end

%% ---------------- 6. Output video ----------------
if writeOutput
    vw = VideoWriter('dmd_result.mp4', 'MPEG-4'); vw.FrameRate = fps; vw.Quality = 100; open(vw);
    for j = 1:K
        row = [tile(V(:,j)), tile(Xbg(:,j)), ...
               tile(abs(Xfg(:,j)))/fgScale, tile(abs(Xband(:,j)))/bnScale];
        row = min(max(row, 0), 1);
        writeVideo(vw, upFrame(repmat(row, [1 1 3]), videoUpscale));
    end
    close(vw);
    disp('Wrote dmd_result.mp4  [original | background | foreground | band]');
end

if writeOutput && colorOutput
    vc = VideoWriter('dmd_result_color.mp4', 'MPEG-4'); vc.FrameRate = fps; vc.Quality = 100; open(vc);
    for j = 1:K
        rgbj = double(Vrgb(:,:,:,j))/255;
        [fgC, bandC] = colorPanels(rgbj, bgRGB, Xfg(:,j), Xband(:,j), fgScale, bnScale, h, w, heat);
        writeVideo(vc, upFrame(min(max([rgbj, bgRGB, fgC, bandC], 0), 1), videoUpscale));
    end
    close(vc);
    disp('Wrote dmd_result_color.mp4  [original | background | foreground | band heat overlay]');
end

%% ---------------- 6c. High-quality colour video (analysis stays small, display is sharp) ----------------
% The DMD runs on small frames (memory!). For viewing, the real full-size colour frames are
% re-read and the small background/foreground/band masks are enlarged and applied to them.
if writeOutput && hiResVideo && isfile(videoFile)
    vr2  = VideoReader(videoFile);
    pick = unique(round(linspace(1, K, min(K, 40))));          % ~40 frames for the sharp background
    stackBg = [];  got = 0;  k2 = 0;  idx2 = 0;
    while hasFrame(vr2) && k2 < K
        f2 = readFrame(vr2);  idx2 = idx2 + 1;
        if mod(idx2-1, frameSkip) ~= 0, continue; end
        k2 = k2 + 1;
        if any(pick == k2)
            f2 = imresize(f2, displayScale);
            if size(f2,3) == 1, f2 = repmat(f2, [1 1 3]); end
            got = got + 1;
            if got == 1, stackBg = zeros([size(f2) numel(pick)], 'uint8'); end
            stackBg(:,:,:,got) = f2;
        end
    end
    stackBg = stackBg(:,:,:,1:got);
    [H2, W2, ~] = size(stackBg(:,:,:,1));
    bgHi = zeros(H2, W2, 3);
    for c = 1:3
        Mc = reshape(double(stackBg(:,:,c,:))/255, H2*W2, got);
        bgHi(:,:,c) = reshape(staticImage(Mc, bgStat, bgQuantile), H2, W2);
    end
    clear stackBg

    vr2 = VideoReader(videoFile);
    vh  = VideoWriter('dmd_result_hq.mp4', 'MPEG-4'); vh.FrameRate = fps; vh.Quality = 95; open(vh);
    k2 = 0;  idx2 = 0;
    while hasFrame(vr2) && k2 < K
        f2 = readFrame(vr2);  idx2 = idx2 + 1;
        if mod(idx2-1, frameSkip) ~= 0, continue; end
        k2 = k2 + 1;
        f2 = imresize(f2, [H2 W2]);
        if size(f2,3) == 1, f2 = repmat(f2, [1 1 3]); end
        rgbHi = double(f2)/255;
        mfg   = imresize(abs(reshape(Xfg(:,k2),   h, w)), [H2 W2], 'bicubic');
        mband = imresize(abs(reshape(Xband(:,k2), h, w)), [H2 W2], 'bicubic');
        [fgH, bandH] = colorPanelsHi(rgbHi, bgHi, mfg, mband, fgScale, bnScale, hot(256));
        writeVideo(vh, upFrame(min(max([rgbHi, bgHi, fgH, bandH], 0), 1), 1));
    end
    close(vh);
    disp('Wrote dmd_result_hq.mp4  [sharp: original | background | foreground | band heat overlay]');
end

%% ---------------- 6b. Play the whole result as a movie ----------------
if playVideo
    figure('Name','Playback (all frames)', 'Position', [80 80 1100 330]);
    ax = axes('Position', [0 0 1 0.88]);
    for j = 1:K
        if colorOutput
            rgbj = double(Vrgb(:,:,:,j))/255;
            [fgC, bandC] = colorPanels(rgbj, bgRGB, Xfg(:,j), Xband(:,j), fgScale, bnScale, h, w, heat);
            frame = [rgbj, bgRGB, fgC, bandC];
        else
            row = [tile(V(:,j)), tile(Xbg(:,j)), tile(abs(Xfg(:,j)))/fgScale, tile(abs(Xband(:,j)))/bnScale];
            frame = repmat(row, [1 1 3]);
        end
        frame = min(max(frame, 0), 1);
        if j == 1
            hIm = imshow(frame, 'Parent', ax);
        else
            set(hIm, 'CData', frame);
        end
        title(ax, sprintf('original | background | foreground | band      t = %.2f s  (frame %d / %d)', tvec(j), j, K));
        drawnow;
        pause(1/fps);
    end
end

%% ---------------- 7. Ground truth: track the moving object directly ----------------
if doTracking
    % Independent check of the swing frequency and damping (no DMD involved).
    Fm = max(V - Vm, 0);                               % bright moving part (ball + string)
    [xg, ~] = meshgrid(1:w, 1:h);
    cx = (xg(:)' * Fm) ./ (sum(Fm,1) + eps);           % centroid x of the moving part, per frame
    cs = movmean(cx - mean(cx), 9);                    % ~0.3 s smoothing
    tp = find(diff(sign(diff(cs))) ~= 0) + 1;          % turning points = left/right extremes
    nTurn  = numel(tp);
    fTurn  = (nTurn/2) / tvec(end);                    % 2 turning points per full swing
    win    = 0.5 - 0.5*cos(2*pi*(0:K-1)'/(K-1));
    Nf     = 8 * 2^nextpow2(K);
    Yc     = abs(fft(cs(:) .* win, Nf));
    fa     = (0:Nf/2) * fps / Nf;
    [~, im] = max(Yc(2:Nf/2+1));  fFFT = fa(im+1);
    if nTurn >= 3
        pf = polyfit(tvec(tp)', log(abs(cs(tp))' + eps), 1);   % envelope ~ exp(-a t)
        decay = -pf(1);
    else
        decay = NaN;
    end
    fprintf('\n--- Direct tracking of the moving object ---\n');
    fprintf('Turning points found: %d  ->  swing frequency %.3f Hz (period %.2f s)\n', nTurn, fTurn, 1/fTurn);
    fprintf('FFT peak of centroid: %.3f Hz\n', fFFT);
    fprintf('Envelope decay rate: %.3f 1/s (amplitude left at end of clip: %.0f %%)\n', decay, 100*exp(-decay*tvec(end)));

    figure('Name','Object tracking');
    plot(tvec, cs, 'b-'); hold on; plot(tvec(tp), cs(tp), 'ro'); grid on;
    xlabel('Time (s)'); ylabel('Centroid x (px, mean removed)');
    title(sprintf('Tracked swing: %d turning points, f = %.2f Hz', nTurn, fTurn));
end

%% ---------------- Local functions ----------------
function [L, S] = rpca_ialm(M, lambda, tol, maxIter)
% Robust PCA by inexact augmented Lagrange multipliers: M = L + S
    [m, n] = size(M);
    normM = norm(M, 'fro');
    nrm2  = norm(M, 2);
    Y  = M / max(nrm2, norm(M(:), inf)/lambda);
    mu = 1.25 / nrm2;  muBar = mu * 1e7;  rho = 1.5;
    S = zeros(m, n);  L = zeros(m, n);
    for it = 1:maxIter
        T = M - S + Y/mu;
        [U, Sg, Vv] = svd(T, 'econ');
        sv  = diag(Sg);
        svp = sum(sv > 1/mu);
        L = U(:,1:svp) * diag(sv(1:svp) - 1/mu) * Vv(:,1:svp)';
        T = M - L + Y/mu;
        S = sign(T) .* max(abs(T) - lambda/mu, 0);
        Z = M - L - S;
        Y = Y + mu*Z;
        mu = min(mu*rho, muBar);
        if norm(Z, 'fro')/normM < tol, break; end
    end
end

function Mv = staticImage(M, stat, q)
% Per-pixel static image over time: 'quantile', 'median' or 'mean'
    K = size(M, 2);
    switch stat
        case 'quantile'
            Ms = sort(M, 2);
            Mv = Ms(:, min(K, max(1, round(q*K))));
        case 'median'
            Mv = median(M, 2);
        otherwise
            Mv = mean(M, 2);
    end
end

function [fgC, bandC] = colorPanels(rgbj, bgRGB, xfg, xmot, fgScale, bnScale, h, w, cmap)
% fgC   : the real colours of the frame, kept only where the moving part is strong
% bandC : dimmed colour background + heat-map of the selected frequency band
    m   = min(abs(reshape(xfg, h, w)) / (0.5*fgScale), 1) .^ 0.7;
    fgC = rgbj .* repmat(m, [1 1 3]);
    b   = min(abs(reshape(xmot, h, w)) / bnScale, 1) .^ 0.6;
    heatImg = reshape(cmap(round(b(:)*255) + 1, :), h, w, 3);   % colour-map lookup
    bandC = min(0.5*bgRGB + heatImg, 1);
end

function F = upFrame(F, s)
% Enlarge a frame by pixel replication (no toolbox) and keep even width/height for MPEG-4
    if s > 1, F = repelem(F, s, s, 1); end
    F = F(1:2*floor(size(F,1)/2), 1:2*floor(size(F,2)/2), :);
end

function [fgC, bandC] = colorPanelsHi(rgbHi, bgHi, mfg, mband, fgScale, bnScale, cmap)
% Same as colorPanels, but the masks are already enlarged to the sharp frame size
    m   = min(max(mfg, 0) / (0.5*fgScale), 1) .^ 0.7;
    fgC = rgbHi .* repmat(m, [1 1 3]);
    b   = min(max(mband, 0) / bnScale, 1) .^ 0.6;
    heatImg = reshape(cmap(round(b(:)*255) + 1, :), size(b,1), size(b,2), 3);
    bandC = min(0.5*bgHi + heatImg, 1);
end
