from parsl.app.app import python_app, bash_app
import parsl_utils


# PARSL APPS:
# SOURCES:
# - https://github.com/inkawhich/synthetic-to-measured-sar

@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['compute_partition'])
def prepare_rundir(run_dir, data_repo_dir="./SAMPLE_Public_Dist_A", inputs = [], stdout= 'std.out', stderr = 'std.err'):
    return '''
        cd {run_dir}
        if ! [ -d "{data_repo_dir}" ]; then
            git clone https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public {data_repo_dir}
        fi
    '''.format(
        run_dir = run_dir,
        data_repo_dir = data_repo_dir
    )


# Main training file for SAMPLE Experiment 4.1, where we vary the percentage of synthetic vs measured
# data in the training set

# Defaults set to experiment 4.1 specifics

#AT_EPS = 2./255.; AT_ALPHA = 0.5/255.; AT_ITERS = 7
#AT_EPS = 4./255.; AT_ALPHA = 1./255. ; AT_ITERS = 7
#AT_EPS = 8./255.; AT_ALPHA = 2./255. ; AT_ITERS = 7
@parsl_utils.parsl_wrappers.log_app
@python_app(executors=['compute_partition'])
def train(ITER, K=0.0, dataset_root=["./SAMPLE_Public_Dist_A/png_images/qpm"], DSIZE=64, num_epochs=60, batch_size=128,
          learning_rate_decay_schedule=[61], learning_rate=0.001, gamma=0.1, weight_decay=0., dropout=0.4, gaussian_std=0.4,
          uniform_range=0., simClutter=0., flipProb=0., degrees=0, LBLSMOOTHING_PARAM=0.1, MIXUP_ALPHA=0.1, std = 'std.out', inputs = []):

    # Parsl requires you to load all inside the function!
    import numpy as np
    import os
    import random
    import torch
    import torch.nn as nn
    import torch.utils.data as utilsdata
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib
    # Fails: matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import scipy.stats as st
    from torch.autograd import Variable

    # STDOUT and STDERR produced by Python apps remotely are not captured.
    # https://parsl.readthedocs.io/en/stable/userguide/apps.html#python-apps -> Limitations
    std_f = open(std, 'w')

    # Custom
    import sys
    sys.path.append(os.path.join(os.getcwd(),'models/pytorch'))
    import models
    import create_split
    import Dataset_fromPythonList as custom_dset
    import helpers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Normalization Constants for range [-1,+1]
    MEAN = torch.tensor([0.5], dtype=torch.float32).view([1, 1, 1]).to(device)
    STD = torch.tensor([0.5], dtype=torch.float32).view([1, 1, 1]).to(device)

    #################################################################################################################
    # Load Model
    #################################################################################################################
    net = None
    #net = models.sample_model(num_classes=10, drop_prob=dropout).to(device)
    net = models.resnet18(num_classes=10, drop_prob=dropout).to(device)
    #net = models.wide_resnet18(num_classes=10, drop_prob=dropout).to(device);
    net.train()
    std_f.write(str(net)); std_f.write('\n'); std_f.flush()
    # Optimizer
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Build the checkpoint prefix for this run
    #checkpoint_prefix = "tmp-checkpoints/{}_{}_seed{}_{}".format(model_name,dataset_name,seed,perturbation_method)

    # Define the learning rate schedule
    learning_rate_table = helpers.create_learning_rate_table(
        learning_rate, learning_rate_decay_schedule, gamma, num_epochs)

    #################################################################################################################
    # Create datasets
    #################################################################################################################

    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(DSIZE),
        transforms.RandomRotation(degrees),
        transforms.RandomHorizontalFlip(flipProb),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(DSIZE),
        transforms.ToTensor(),
    ])

    # Create the measured/synthetic split training and test data
    full_train_list, test_list = create_split.create_mixed_dataset_exp41(
        dataset_root, K)

    # Create validation set split
    val_set_size = 0  # int(0.15 * len(full_train_list))
    val_sample_inds = random.sample(
        list(range(len(full_train_list))), val_set_size)
    train_list = []
    val_list = []
    for ind in range(len(full_train_list)):
        if ind in val_sample_inds:
            val_list.append(full_train_list[ind])
        else:
            train_list.append(full_train_list[ind])

    std_f.write("# Train: " + str(len(train_list)) + "\n")
    std_f.write("# Val:	  " + str(len(val_list)) + "\n")
    std_f.write("# Test:  " + str(len(test_list)) + "\n")
    std_f.flush()

    # Construct datasets and dataloaders
    trainset = custom_dset.Dataset_fromPythonList(
        train_list, transform=transform_train)
    trainloader = utilsdata.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, timeout=1000)
    #valset = custom_dset.Dataset_fromPythonList(val_list, transform=transform_test)
    #valloader = utilsdata.DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=2,timeout=1000)
    testset = custom_dset.Dataset_fromPythonList(
        test_list, transform=transform_test)
    testloader = utilsdata.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2, timeout=1000)

    #################################################################################################################
    # Training Loop
    #################################################################################################################

    global_training_iteration = 0.
    best_test_acc = 0.
    final_test_acc = 0.
    final_train_acc = 0.

    for epoch in range(num_epochs):

        # Decay learning rate according to decay schedule
        helpers.adjust_learning_rate(optimizer, epoch, learning_rate_table)
        net.train()
        std_f.write("Starting Epoch {}/{}. lr = {}\n".format(epoch,
              num_epochs, learning_rate_table[epoch]))
        std_f.flush()

        running_correct = 0.
        running_total = 0.
        running_loss_sum = 0.
        running_real_cnt = 0.

        for batch_idx, (data, labels, pth) in enumerate(trainloader):
            data = data.to(device)
            labels = labels.to(device)

            # MIXUP
            #mixed_data, targets_a, targets_b, lam = helpers.mixup_data(data, labels, MIXUP_ALPHA, use_cuda=True)
            #mixed_data, targets_a, targets_b      = map(Variable, (mixed_data, targets_a, targets_b))

            # Gaussian/Uniform/SimClutter Noise
            if (uniform_range != 0):
                noise = (torch.rand_like(data)-.5)*2*uniform_range
                data += noise
                data = torch.clamp(data, 0, 1)
            if (gaussian_std != 0):
                data += torch.randn_like(data)*gaussian_std
                data = torch.clamp(data, 0, 1)
                #mixed_data += torch.randn_like(mixed_data)*gaussian_std;
                #mixed_data = torch.clamp(mixed_data, 0, 1);
            if (simClutter != 0):
                data = helpers.SimClutter_attack(device, data, simClutter)

            # ADVERSARIALLY PERTURB DATA
            #data = helpers.PGD_Linf_attack(net, device, data.clone().detach(), labels, eps=AT_EPS, alpha=AT_ALPHA, iters=AT_ITERS)

            # Optional: Plot some training samples
            # plt.figure(figsize=(10,3))
            # plt.subplot(1,6,1);plt.imshow(data[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            # plt.subplot(1,6,2);plt.imshow(data[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            # plt.subplot(1,6,3);plt.imshow(data[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            # plt.subplot(1,6,4);plt.imshow(unshifted[0].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[0].split("/")[-1].split("_")[:2])
            # plt.subplot(1,6,5);plt.imshow(unshifted[1].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[1].split("/")[-1].split("_")[:2])
            # plt.subplot(1,6,6);plt.imshow(unshifted[2].cpu().numpy().squeeze(),vmin=0,vmax=1);plt.title(pth[2].split("/")[-1].split("_")[:2])
            # plt.show()
            # exit()

            # MIXUP
            #outputs = net((mixed_data-MEAN)/STD)
            #loss = helpers.mixup_criterion(nn.CrossEntropyLoss(), outputs, targets_a, targets_b, lam)

            # Forward pass data through model. Normalize before forward pass
            outputs = net((data-MEAN)/STD)

            # VANILLA CROSS-ENTROPY
            loss = F.cross_entropy(outputs, labels)

            # LABEL SMOOTHING LOSS
            #sl = helpers.smooth_one_hot(labels,10,smoothing=LBLSMOOTHING_PARAM)
            #loss =  helpers.xent_with_soft_targets(outputs, sl)

            # COSINE LOSS
            #one_hots = smooth_one_hot(labels,10,smoothing=0.)
            #loss = (1. - (one_hots * F.normalize(outputs,p=2,dim=1)).sum(1)).mean()

            # Calculate gradient and update parameters
            optimizer.zero_grad()
            net.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)	# For cosine loss
            optimizer.step()

            # Measure accuracy and loss for this batch
            _, preds = outputs.max(1)
            running_total += labels.size(0)
            running_correct += preds.eq(labels).sum().item()
            # running_correct += (lam * preds.eq(targets_a.data).cpu().sum().float() + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) # For Mixup
            running_loss_sum += loss.item()

            # Global training iteration count across epochs
            global_training_iteration += 1

            # Compute measured/synthetic split for the batch
            for tp in pth:
                if "/real/" in tp:
                    running_real_cnt += 1.

        # End of epoch - print stats
        std_f.write("[{}] Epoch [ {} / {} ]; lr: {} TrainAccuracy: {} TrainLoss: {} %-Real: {}\n".format(ITER, epoch, num_epochs,
              learning_rate_table[epoch], running_correct/running_total, running_loss_sum/running_total, running_real_cnt/running_total))
        #val_acc,val_loss = helpers.test_model(net,device,valloader,MEAN,STD)
        #print("\t[{}] Epoch [ {} / {} ]; ValAccuracy: {} ValLoss: {}".format(ITER,epoch,num_epochs,val_acc,val_loss))

        test_acc, test_loss = helpers.test_model(
            net, device, testloader, MEAN, STD)

        std_f.write("\t[{}] Epoch [ {} / {} ]; TestAccuracy: {} TestLoss: {}\n".format(ITER,
              epoch, num_epochs, test_acc, test_loss))
        if test_acc > best_test_acc:
            std_f.write("\tNew best test accuracy!\n")
            best_test_acc = test_acc
        std_f.flush()
        final_test_acc = test_acc
        final_train_acc = running_correct/running_total

    # Optional: Save a model checkpoint here
    # if final_test_acc > SAVE_THRESH:
    #	helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, SAVE_CKPT)
    #	exit("Found above average model to save. Exit now!")
    #helpers.save_checkpoint({'test_acc': final_test_acc,'state_dict': net.state_dict()}, False, "{}_K{}_ITER{}".format(SAVE_CKPT,int(100*K),ITER))

    if final_train_acc > .5:
        std_f.write("BREAK. FINAL RECORDED TEST ACC = " + str(final_test_acc) + "\n")
        return final_test_acc
    else:
        std_f.write("MODEL NEVER LEARNED ANYTHING. NOT RECORDING\n")

    std_f.close()


@parsl_utils.parsl_wrappers.log_app
@python_app(executors=['compute_partition'])
def merge(K, REPEAT_ITERS, inputs = []):
    import numpy as np

    ACCUMULATED_ACCURACIES = [i for i in inputs if i]
    minacc = np.array(ACCUMULATED_ACCURACIES).min()
    maxacc = np.array(ACCUMULATED_ACCURACIES).max()
    avgacc = np.array(ACCUMULATED_ACCURACIES).mean()
    stdacc = np.array(ACCUMULATED_ACCURACIES).std()
    lenacc = len(ACCUMULATED_ACCURACIES)
    return ACCUMULATED_ACCURACIES, minacc, maxacc, avgacc, stdacc, lenacc


@parsl_utils.parsl_wrappers.log_app
@python_app(executors=['compute_partition'])
def preprocess_images(angle, dataset_root, out_dir, inputs = []):
    import glob
    import os
    from PIL import Image

    def rotate_image(angle, input_path, output_path):
        # Load the image
        image = Image.open(input_path)
        # Rotate the image by 90 degrees
        rotated_image = image.rotate(int(angle))
        # Save the rotated image to disk
        rotated_image.save(output_path)

    for case in ['real', 'synth']:
        out_dir_case = os.path.join(out_dir, case)
        os.makedirs(out_dir_case, exist_ok=True)
        [
            rotate_image(
                angle,
                img_path,
                os.path.join(
                    out_dir_case,
                    os.path.basename(img_path).replace('.png', '_' + str(angle) + '.png')
                )
            )
            for img_path in glob.glob("{}/{}/*/*.png".format(dataset_root, case))
        ] 



@parsl_utils.parsl_wrappers.log_app
@bash_app(executors=['compute_partition'])
def preprocess_images_matlab(angle, dataset_root, out_dir, inputs = []):
    return '''
    set -x
    date
    mkdir -p {out_dir}

    '''
