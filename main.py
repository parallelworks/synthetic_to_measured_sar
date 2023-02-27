import sys, os, json, time
from random import randint
import argparse
from functools import partial


import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app

import numpy as np

import parsl_utils
from parsl_utils.config import config, exec_conf, pwargs, job_number
from parsl_utils.data_provider import PWFile


from workflow_apps import prepare_rundir, mpi_add_noise, train, preprocess_images_python, merge, preprocess_images_matlab

if __name__ == '__main__':
    REPEAT_ITERS = int(pwargs['REPEAT_ITERS'])
    K = float(pwargs['K'])

    data_repo_dir = pwargs['data_repo_dir']
    dataset_root = os.path.join(data_repo_dir, "png_images/qpm")

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    print("\n\n**********************************************************")
    print("Preparing Run Directory")
    print("**********************************************************")
    prepare_rundir_fut = prepare_rundir(
        exec_conf['compute_partition']['RUN_DIR'],
        data_repo_dir = data_repo_dir,
        inputs = [ 
            PWFile(
                url = 'file://usercontainer/{cwd}/models/'.format(cwd = os.getcwd()),
                local_path = '{remote_dir}/models'.format(remote_dir =  exec_conf['compute_partition']['RUN_DIR'])
            )
        ],
        stdout = 'prepare_rundir_fut.out',
        stderr = 'prepare_rundir_fut.err'
    )

    print("\n\n**********************************************************")
    print("Preprocessing Images")
    print("**********************************************************")
    if pwargs['prepro_tool'] == 'matlab':
        # FIXME: Generalize internal_ip_controller 
        preprocess_images = partial(
            preprocess_images_matlab,
            matlab_bin = pwargs['matlab_bin'],
            matlab_server_port = pwargs['matlab_server_port'],
            matlab_daemon_port = pwargs['matlab_daemon_port'],
            internal_ip_controller = config.executors[0].address
        )
    else:
        preprocess_images = preprocess_images_python

    # Add noise to the training data and rotate it
    pp_futs = []
    pp_images_out_dir = 'noise_rotated_images'
    noise_images_futs = []
    noise_images_out_dir = 'noise_images'
    for case in ['real', 'synth']:
        # Directory with original images
        src_dir = os.path.join(dataset_root, case)
        # Directory with noisy images
        noise_dir = os.path.join(noise_images_out_dir, case)
        # Directory with noisy rotated images         
        noise_rot_dir = os.path.join(pp_images_out_dir, case)

        mpi_fut = mpi_add_noise(
            pwargs['np'],
            src_dir, 
            noise_dir,
            pwargs['noise_amount'],
            stdout = './std-' + case + '-mpi-noise.out',
            stderr = './std-' + case + '-mpi-noise.err',
            inputs = [prepare_rundir_fut]
        )
        
        for rot_angle in pwargs['rot_angles'].split('___'):
            pp_futs.append(
                preprocess_images(
                    rot_angle, 
                    noise_dir, 
                    noise_rot_dir,
                    stdout = './std-' + case + '-' + rot_angle + '.out',
                    stderr = './std-' + case + '-' + rot_angle + '.err',
                    inputs = [mpi_fut]
                )
            )

    ACCUMULATED_ACCURACIES_FUTS = []
    print("\n\n**********************************************************")
    print("Loop Over Training Runs to Get Average Accuracies")
    print("**********************************************************")
    for ITER in range(REPEAT_ITERS):
        print("**********************************************************")
        print("Starting Iter: {} / {} for K = {}".format(ITER, REPEAT_ITERS, K))
        print("**********************************************************")

        ACCUMULATED_ACCURACIES_FUTS.append(
            train(
                ITER,
                K = K,
                DSIZE = int(pwargs['DSIZE']),
                num_epochs = int(pwargs['num_epochs']),
                batch_size = int(pwargs['batch_size']),
                learning_rate_decay_schedule = [ int(lr) for lr in pwargs['learning_rate_decay_schedule'].split('---') ],
                learning_rate = float(pwargs['learning_rate']),
                gamma = float(pwargs['gamma']),
                weight_decay = float(pwargs['weight_decay']),
                dropout = float(pwargs['dropout']),
                gaussian_std = float(pwargs['gaussian_std']),
                uniform_range = float(pwargs['uniform_range']),
                simClutter = float(pwargs['simClutter']),
                flipProb = float(pwargs['flipProb']),
                degrees = int(pwargs['degrees']),
                LBLSMOOTHING_PARAM = float(pwargs['LBLSMOOTHING_PARAM']),
                MIXUP_ALPHA = float(pwargs['MIXUP_ALPHA']),
                dataset_root = [dataset_root, pp_images_out_dir],
                std = 'std-{}.out'.format(ITER),
                inputs = pp_futs
            )
        )


    merge_fut = merge(K, REPEAT_ITERS, inputs = ACCUMULATED_ACCURACIES_FUTS)
    ACCUMULATED_ACCURACIES, minacc, maxacc, avgacc, stdacc, lenacc = merge_fut.result()

    print("\n\nEND OF TRAINING!")
    print("ACCUMULATED ACCURACIES: ", ACCUMULATED_ACCURACIES)
    print("\tMin = ", minacc)
    print("\tMax = ", maxacc)
    print("\tAvg = ", avgacc)
    print("\tStd = ", stdacc)
    print("\tlen = ", lenacc)
