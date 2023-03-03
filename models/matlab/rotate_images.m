% Read the source directory from an environment variable
src_dir = getenv('SRC_DIR');

% Read the destination directory from an environment variable
dst_dir = getenv('DST_DIR');

% Read the rotation angle from an environment variable and convert it to a number
angle = str2double(getenv('ANGLE'));

% Get a list of all PNG files in the source directory
files = dir(fullfile(src_dir, '**', '*.png'));

% Loop over each file and process it
for i = 1:numel(files)
   % Load the image
   img_path = fullfile(files(i).folder, files(i).name);
   disp(img_path);
   img_path_out = strrep(img_path, src_dir, dst_dir);
   img_path_out = strrep(img_path_out, '.png', strcat('_rotangle_', num2str(angle),'.png'));

   disp(img_path_out)
   [img_dir_out, ~, ~] = fileparts(img_path_out);
   mkdir(img_dir_out)

   img = imread(img_path);

   % Rotate the image by the given angle (in degrees)
   if angle == 90
       img_rotated = rot90(img, -1);
   elseif angle == 180
       img_rotated = rot90(img, 2);
   else
       img_rotated = rot90(img, 1);
   end

   % Save the rotated image to the destination directory
   imwrite(img_rotated, img_path_out);
end