import os, sys
import glob
import random

from PIL import Image
from mpi4py import MPI

def split_list(lst, n):
    '''
    Split a list into n sublists of similar sizes
    '''
    k, m = divmod(len(lst), n)
    return (lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))

def add_noise_to_image(file_path, src_dir, dst_dir, noise_amount = 25):
    # Open the image file
    with Image.open(file_path) as img:
        # Add random noise to the image
        noisy_data = bytes([min(255, max(0, p + random.randint(-noise_amount, noise_amount))) for p in img.tobytes()])
        noisy_img = Image.frombytes(img.mode, img.size, noisy_data)

        # Add _noise to the file name and save the noisy image
        noisy_file_path = file_path.replace(src_dir, dst_dir).replace('.png', '_noise.png')
        os.makedirs(os.path.dirname(noisy_file_path), exist_ok=True)
        noisy_img.save(noisy_file_path)

#dataset_root = './SAMPLE_dataset_public/png_images/qpm' 
src_dir = sys.argv[1] #os.path.join(dataset_root, 'real')
dst_dir = sys.argv[2] #'noise_images/real'
noise_amount = int(sys.argv[3])

comm = MPI.COMM_WORLD
sendbuf = []

if comm.rank == 0:
    img_paths = glob.glob("{}/*/*.png".format(src_dir))
    # Split list into np ranks
    sendbuf = split_list(img_paths, comm.size)

# Send each rank a sublist of image paths
img_paths = comm.scatter(sendbuf, root = 0)

for img_path in img_paths:
    add_noise_to_image(img_path, src_dir, dst_dir, noise_amount = noise_amount)

