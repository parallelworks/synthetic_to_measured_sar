% Specify the source directory containing the PNG images
src_dir = '__src_dir__';

% Specify the destination directory where the rotated images will be saved
dst_dir = '__dst_dir__';

% Rotation angle:
angle = str2num('__angle__');

% Get a list of all PNG files in the source directory
files = dir(strcat(src_dir, '/*/*.png'))
% Loop over each file and process it
for i = 1:numel(files)
   % Load the image
   img_path = fullfile(files(i).folder, files(i).name);
   disp(img_path);
   img_path_out = strrep(img_path, src_dir, dst_dir)
   img_path_out = strrep(img_path_out, '.png', strcat('_rotangle_', num2str(angle),'.png'))

   disp(img_path_out)
   [img_dir_out, ~, ~] = fileparts(img_path_out);
   mkdir(img_dir_out)

   img = imread(img_path);

   % Rotate the image by the given angle (in degrees)
   %img_rotated = imrotate(img, angle);
   if angle == '90'
       img_rotated = rot90(img, -1);
   elseif angle == '180'
       img_rotated = rot90(img, 2);
   else
       img_rotated = rot90(img, 1);
   end

    
   % Save the rotated image to the destination directory
   imwrite(img_rotated, img_path_out);
end