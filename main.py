import os
from functools import partial

import parsl
print(parsl.__version__, flush = True)

import parsl_utils
from parsl_utils.config import config, form_inputs, executor_dict #exec_conf
from parsl_utils.data_provider import PWFile


from workflow_apps import prepare_rundir, mpi_add_noise, train, preprocess_images_python, merge, preprocess_images_matlab, preprocess_images_cmatlab

if __name__ == '__main__':
    REPEAT_ITERS = int(form_inputs['cnn']['REPEAT_ITERS'])
    K = float(form_inputs['cnn']['K'])

    data_repo_dir = form_inputs['data']['repo_dir']
    dataset_root = os.path.join(data_repo_dir, "png_images/qpm")

    print('Loading Parsl Config', flush = True)
    parsl.load(config)

    print("\n\n**********************************************************")
    print("Preparing Run Directory")
    print("**********************************************************")
    prepare_rundir_fut = prepare_rundir(
        data_repo_dir = data_repo_dir,
        inputs = [ 
            PWFile(
                url = './models/',
                local_path = './models/'
            )
        ],
        stdout = 'prepare_rundir_fut.out',
        stderr = 'prepare_rundir_fut.err'
    )

    print("\n\n**********************************************************")
    print("Preprocessing Images")
    print("**********************************************************")
    if form_inputs['prepro']['tool'] == 'matlab':
        # FIXME: Generalize internal_ip_controller 
        preprocess_images = partial(
            preprocess_images_matlab,
            matlab_bin = form_inputs['prepro']['matlab_bin'],
            matlab_server_port = form_inputs['prepro']['matlab_server_port'],
            matlab_daemon_port = form_inputs['prepro']['matlab_daemon_port'],
            internal_ip_controller = config.executors[0].address
        )
    elif form_inputs['prepro']['tool'] == 'matlab_compiled':
        preprocess_images = partial(
            preprocess_images_cmatlab,
            mcrroot = form_inputs['prepro']['mcrroot']
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
            form_inputs['mpi']['np'],
            src_dir, 
            noise_dir,
            form_inputs['mpi']['noise_amount'],
            stdout = './std-' + case + '-mpi-noise.out',
            stderr = './std-' + case + '-mpi-noise.err',
            inputs = [prepare_rundir_fut]
        )
        
        for rot_angle in form_inputs['prepro']['rot_angles'].split('___'):
            pp_fut = preprocess_images(
                rot_angle, 
                noise_dir, 
                noise_rot_dir,
                stdout = './std-' + case + '-' + rot_angle + '.out',
                stderr = './std-' + case + '-' + rot_angle + '.err',
                inputs = [mpi_fut]
            )
            pp_futs.append(pp_fut)


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
                DSIZE = int(form_inputs['cnn']['DSIZE']),
                num_epochs = int(form_inputs['cnn']['num_epochs']),
                batch_size = int(form_inputs['cnn']['batch_size']),
                learning_rate_decay_schedule = [ int(lr) for lr in form_inputs['cnn']['learning_rate_decay_schedule'].split('---') ],
                learning_rate = float(form_inputs['cnn']['learning_rate']),
                gamma = float(form_inputs['cnn']['gamma']),
                weight_decay = float(form_inputs['cnn']['weight_decay']),
                dropout = float(form_inputs['cnn']['dropout']),
                gaussian_std = float(form_inputs['cnn']['gaussian_std']),
                uniform_range = float(form_inputs['cnn']['uniform_range']),
                simClutter = float(form_inputs['cnn']['simClutter']),
                flipProb = float(form_inputs['cnn']['flipProb']),
                degrees = int(form_inputs['cnn']['degrees']),
                LBLSMOOTHING_PARAM = float(form_inputs['cnn']['LBLSMOOTHING_PARAM']),
                MIXUP_ALPHA = float(form_inputs['cnn']['MIXUP_ALPHA']),
                dataset_root = [dataset_root, pp_images_out_dir],
                std = 'train-{}.out'.format(ITER),
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
