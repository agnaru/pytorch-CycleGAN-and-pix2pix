"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import os
import pickle
import numpy as np
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from data.base_dataset import get_transform, get_custom_transform
from data.image_folder import make_dataset
from models import create_model
from PIL import Image
from util.visualizer import save_images
from util import util
from util.visualizer import Visualizer

import wandb

def getExG(img):
    '''
    img -> ExG tonal img
    '''
    R = img[..., 0]/255.
    G = img[..., 1]/255.
    B = img[..., 2]/255.
    tonal_img = 2 * G - R - B
    
    return tonal_img

def getExGmask(img, threshold = 0):
    '''
    RGB : 1 ~ 255 uint8
    threshold for CultiEnv
    '''
    tonal_img = getExG(img)
    mask = tonal_img > threshold
    return mask

def applyExG(img, threshold = 0):
    '''
    RGB : 1 ~ 255 uint8
    threshold for CultiEnv
    '''
    tonal_img = getExG(img)
    mask = tonal_img > threshold
    green = np.zeros_like(img, np.uint8)
    green[mask] = img[mask]
    return green

def evaluation(model, dataset, opt, input_nc, output_nc, cnt = False):

    transform_A = get_custom_transform(opt, grayscale=(input_nc == 1))
    
    transfered_data = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.B2A_set_input(data)  # unpack data from data loader
        model.B2A_test()           # run inference
        transfered_data.append(util.tensor2im(model.fake_A))
    
    dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
    A_paths = sorted(make_dataset(dir_A, opt.max_dataset_size))
    
    day_data =[]
    for j, A_path in enumerate(A_paths):
        A_img = Image.open(A_path).convert('RGB')
        A = transform_A(A_img)
        day_image_numpy = A.data.cpu().float().numpy()
        day_image_numpy = (np.transpose(day_image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        day_image_numpy = day_image_numpy.astype(np.uint8)
        day_data.append(day_image_numpy)
    
    day_data = np.array(day_data)
    transfered_data = np.array(transfered_data)
    
    if cnt:
        metric = np.sqrt(np.mean((np.count_nonzero(getExGmask(day_data), axis = (1, 2))/(64*64) - np.count_nonzero(getExGmask(transfered_data), axis = (1, 2))/(64*64))**2))
        
    else:
        metric = np.sqrt(np.mean((applyExG(day_data) - applyExG(transfered_data))**2))
        
    return metric
        

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    best_eval_rmse = None
    
# ------------------------------------------------------------- eval data
    test_opt = TestOptions().parse()
    test_opt.num_threads = 0   # test code only supports num_threads = 0
    test_opt.batch_size = 1    # test code only supports batch_size = 1
    test_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    test_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    test_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    test_opt.phase = "eval"
    test_opt.dataset_mode = 'b2a'
    eval_dataset = create_dataset(test_opt)  # create a dataset given opt.dataset_mode and other options
    test_opt.num_test = len(eval_dataset)
    
    btoA = test_opt.direction == 'BtoA'
    cnt = True if opt.eval_metric == 'count' else False
    input_nc = test_opt.output_nc if btoA else test_opt.input_nc
    output_nc = test_opt.input_nc if btoA else test_opt.output_nc
    
    print(f"test dataset num is {test_opt.num_test}")

# -------------------------------------------------------------
    not_improved = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        
        # update learning rates in the beginning of every epoch.
        
        if epoch != opt.epoch_count:
            model.update_learning_rate()
            visualizer.wandb_run.log({"learning_rate": model.optimizers[0].param_groups[0]['lr']}, step = total_iters)
        else:
            visualizer.wandb_run.log({"learning_rate": model.optimizers[0].param_groups[0]['lr']}, step = total_iters)
            print('learning rate %.7f' % model.optimizers[0].param_groups[0]['lr'])
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0 or opt.use_wandb:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses, total_iters)
            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                eval_rmse = evaluation(model, eval_dataset, test_opt, input_nc, output_nc, cnt)
                visualizer.wandb_run.log({f"eval_rmse_{opt.eval_metric}": eval_rmse}, step = total_iters)
                if best_eval_rmse is None:
                    best_eval_rmse = eval_rmse

                if eval_rmse <= best_eval_rmse:
                    visualizer.wandb_run.summary["best_eval_rmse"] = eval_rmse
                    print('saving the best model (epoch: %d, total_iters: %d). best eval rmse: %f -> %f' % (epoch, total_iters, best_eval_rmse, eval_rmse))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix, opt.use_wandb)
                    best_eval_rmse = eval_rmse
                    not_improved = 0
                    
                elif epoch <= 2:
                    print(f'eval rmse is not improved. best eval rmse: {best_eval_rmse}')
                    
                else:
                    not_improved +=1
                    print(f'eval rmse is not improved. best eval rmse: {best_eval_rmse}')
                    
                if not_improved > 4 and epoch > 2:
                    break

            iter_data_time = time.time()
        
        if not_improved > 4 and epoch > 2:
            break
            
#         if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
#             print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
#             model.save_networks('latest', opt.use_wandb)
#             model.save_networks(epoch, opt.use_wandb)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    if not_improved > 4:
        print('End of epoch %d / %d. Early stopped. best eval rmse %.5f' % (epoch, opt.n_epochs + opt.n_epochs_decay, best_eval_rmse))