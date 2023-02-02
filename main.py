import sys, os, json, time
from random import randint
import argparse

import parsl
print(parsl.__version__, flush = True)
from parsl.app.app import python_app, bash_app

import numpy as np

import parsl_utils
from parsl_utils.config import config, exec_conf, pwargs, job_number
from parsl_utils.data_provider import PWFile


from workflow_apps import prepare_rundir, train, preprocess_images, merge

if __name__ == '__main__':
    angle = 90
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
                url = 'file://usercontainer/{cwd}/models/pytorch/'.format(cwd = os.getcwd()),
                local_path = '{remote_dir}/models/pytorch'.format(remote_dir =  exec_conf['compute_partition']['RUN_DIR'])
            )
        ],
        stdout = 'prepare_rundir_fut.out',
        stderr = 'prepare_rundir_fut.err'
    )

    pp_images_out_dir = 'pp_images'
    preprocess_images_fut = preprocess_images(angle, dataset_root, pp_images_out_dir, inputs = [prepare_rundir_fut])

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
                inputs = [preprocess_images_fut]
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
