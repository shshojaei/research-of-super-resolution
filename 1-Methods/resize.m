%source: https://www.mathworks.com/matlabcentral/answers/245630-resizing-all-images-in-a-folder

%%
function resize()
input_path = 'F:\datasets\Set5\';
output_path = 'F:\datasets\Set5\LRbicx2\';
dc = dir([input_path, '*.png']); % loads all the image infos(name,folder,date,bytes,isdir,datenum)

%loop through all your images to resize
for i = 1:numel(dc) % numel: Number of array elements 
   resize_and_save(dc(i).name,input_path,output_path);
end
end
%%

function[]=resize_and_save(image_name,input_path,output_path)
%load image
image=imread([input_path image_name]);
%resize image
%factor scales: x2,x3,x4 => 0.5, 0.33, 0.25
newimage=imresize(image,0.5);
%save new image
imwrite(newimage,[output_path image_name])
end




