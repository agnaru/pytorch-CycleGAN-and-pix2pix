"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import json
import numpy as np
from options.test_options import TestOptions
from data.base_dataset import get_transform, get_custom_transform
from data.image_folder import make_dataset
from data import create_dataset
from models import create_model
from PIL import Image
from util.visualizer import save_images
from util import util

def saveJSON(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as make_file:
        json.dump(data, make_file, indent="\t")

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


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.phase = "all"
    opt.dataset_mode = 'b2a'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options    

    btoA = opt.direction == 'BtoA'
    input_nc = opt.output_nc if btoA else opt.input_nc
    output_nc = opt.input_nc if btoA else opt.output_nc
    transform_A = get_custom_transform(opt, grayscale=(input_nc == 1))
    dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
    A_paths = sorted(make_dataset(dir_A, opt.max_dataset_size))
    opt.num_test = len(dataset)
    print(f"dataset num is B: {opt.num_test} + A: {len(A_paths)} = {opt.num_test + len(A_paths)}")
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    exg_cnt = {}
    save_path = opt.results_dir
    
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.B2A_set_input(data)  # unpack data from data loader
        model.B2A_test()           # run inference
        img_path = model.get_image_paths()     # get image paths
        if i % 100 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        util.save_image(util.tensor2im(model.fake_A), os.path.join(save_path, img_path[0].split('/')[-1]), aspect_ratio=opt.aspect_ratio)
        # reward calc
#         exg_cnt[img_path[0].split('/')[-1]] = np.count_nonzero(getExGmask(util.tensor2im(model.fake_A))[80:560])/(480*480*3)
    print('pre-processing day images')
    for j, A_path in enumerate(A_paths):
        A_img = Image.open(A_path).convert('RGB')
        A = transform_A(A_img)
        image_numpy = A.data.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)         
        util.save_image(image_numpy, os.path.join(save_path, A_path.split('/')[-1]), aspect_ratio=opt.aspect_ratio)
        if j % 100 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (j, A_path))
        # reward calc
#         exg_cnt[A_path.split('/')[-1]] = np.count_nonzero(getExGmask(image_numpy)[80:560])/(480*480*3)
    
#     print('pre-processing night images')
#     transform_B = get_custom_transform(opt, grayscale=(output_nc == 1))
#     dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
#     B_paths = sorted(make_dataset(dir_B, opt.max_dataset_size))
#     for k, B_path in enumerate(B_paths):
#         B_img = Image.open(B_path).convert('RGB')
#         B = transform_B(B_img)
#         save_path = opt.results_dir
#         image_numpy = B.data.cpu().float().numpy()
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#         image_numpy = image_numpy.astype(np.uint8)
# #         util.save_image(image_numpy, save_path, aspect_ratio=opt.aspect_ratio)
#         if k % 100 == 0:  # save images to an HTML file
#             print('processing (%04d)-th image... %s' % (k, B_path))
#         exg_cnt[B_path.split('/')[-1]] = np.count_nonzero(applyExG(image_numpy)[80:560])/(480*480*3)
        
    saveJSON(exg_cnt, os.path.join(opt.results_dir, "reward.json"))
