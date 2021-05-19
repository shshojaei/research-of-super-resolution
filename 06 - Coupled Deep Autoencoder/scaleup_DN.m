function [imgs, midres] = scaleup_DN(conf, imgs)
% Super-Resolution Iteration
    fprintf('Scale-Up deep network');
    midres = resize(imgs, conf.upsample_factor, conf.interpolate_kernel);
    %interpolated = resize(imgs, conf.scale, conf.interpolate_kernel);
    
    for i = 1:numel(midres)
        testlowinputs = collect(conf, {midres{i}}, conf.upsample_factor, {});
%        testlowinputs = double(testlowinputs)/255;

        % Reconstruct using patches' dictionary and their global projection
        %testhighoutputs=SR_CoupledSAE(testlowinputs);
		if(conf.scale == 3)
			load('model/x3');
		end
		if(conf.scale == 2)
			load('model/x2');
		end
		if(conf.scale == 4)
			load('model/x4');
		end
		m = size(testlowinputs,2);
		a1 = testlowinputs;
		z2 = SRnnl1_W * a1 + repmat(SRnnl1_b,[1, size(a1,2)]);
		a2 =  sigmoid(z2); 
		z3 = SRnnl2_W * a2 + repmat(SRnnl2_b,[1, size(a2,2)]);
		a3 =  sigmoid(z3);
		z4 = SRnnl3_W * a3 + repmat(SRnnl3_b,[1, size(a3,2)]);
		testhighoutputs =  sigmoid(z4);
        % Combine all patches into one image
        img_size = size(imgs{i}) * conf.scale;
        grid = sampling_grid(img_size, ...
            conf.window, conf.overlap, conf.border, conf.scale);
        result = overlap_add(testhighoutputs, img_size, grid);
        imgs{i} = result; % for the next iteration
        fprintf('.');
    end
fprintf('\n');
