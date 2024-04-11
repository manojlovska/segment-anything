import os
import glob
import shutil

path = '/work/anastasija/Materials-Science/segment-anything/output-splitted-t_0.95-prefiltered'
convexity_images = sorted(glob.glob(os.path.join(path, '**/*_convexity.png'),
                         recursive = True))

save_path = '/work/anastasija/Materials-Science/segment-anything/results/CHECK-CONVEXITY/prefiltering-t-0.95'

save_paths = [os.path.join(save_path, im.split('/')[-3], os.path.basename(im)) for im in convexity_images]

for i, im_path in enumerate(convexity_images):
    destination_directory = os.path.join(save_path, im_path.split('/')[-3])
    os.makedirs(destination_directory, exist_ok=True)

    shutil.move(im_path, save_paths[i])









