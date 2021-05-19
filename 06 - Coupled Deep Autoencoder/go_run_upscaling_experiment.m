clear;  
p = pwd;
imgscale = 1; % the scale reference we work with
for upscaling = [2 3 4]; % the magnification factor x2, x3, x4...
image_number = 14; % 5 or 14
input_dir = ['Set' num2str(image_number)]; % Directory with input images from Set5 or Set14 image dataset
pattern = '*.bmp'; % Pattern to process
disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling)]);
disp('We run only for bicubic,SRCDA');

fprintf('\n\n');
tag = [input_dir '_x' num2str(upscaling)];
mat_file = ['conf_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')
        %disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');
    else                            
       
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)
        if conf.scale~=2
            extra = 1;
        else
            extra = 0;
        end
        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';
        conf.model = ['model\x' num2str(conf.scale) '.mat'];
        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [2 2]; % partial overlap (for faster training)
        end
        %startt = tic;
        %conf = learn_dict(conf, load_images(glob('CVPR08-SR/Data/Training', '*.bmp'),extra));       
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)    
        %conf.trainingtime = toc(startt);
        %toc(startt)
        
        save(mat_file, 'conf');                       
        
        % train call        
    end
    
    conf.filenames = glob(input_dir, pattern); % Cell array  
    %conf.filenames = {conf.filenames{4}};
    
    conf.desc = {'Original','Bicubic', 'SRCDA'};
    conf.results = {};
    
    conf.result_dir = qmkdir(['Results-scale' num2str(conf.scale)]);
    conf.result_dirRGB = qmkdir(['ResultsRGB-scale' num2str(conf.scale)]);
  
    %%
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
            
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
        end
        
        res{1} = interpolated; 
        
        startt = tic;
        res{2} = scaleup_DN(conf, low);
        toc(startt)
        conf.countedtime(2,i) = toc(startt); 

           
        result = cat(3, img{1}, interpolated{1}, res{2}{1});
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
            conf.results{i}{j} = fullfile(conf.result_dir, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);            
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    scores = run_comparison(conf,image_number);
    process_scores_Tex(conf, scores,length(conf.filenames));
end