import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torchvision
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import numpy as np
import os
import math
import argparse
import json

import time

from utils.homographies import  get_offsets_from_positions, save_kernels_from_offsets, reblur_offsets



from training_pairs_generation.BlurrySharpPairsDatasetOnline import BlurrySharpPairDatasetOnline


from skimage.io import imsave
import kornia
import traceback



parser = argparse.ArgumentParser(description="Kernel estimator")
parser.add_argument("-s", "--imagesize", type=int, default=-1)
parser.add_argument('-o' ,'--output_dir', help='directory to output the results', default='./training_pairs_generation/sample_training_pairs')
parser.add_argument('-n' ,'--n_positions', type=int, help='number of positions to estimate', default=25)
parser.add_argument('-sf','--sharp_folder', type=str, help='folder with sharp images', required=True)
parser.add_argument('-pf','--positions_folder', type=str, help='folder with camera positions', required=True)
parser.add_argument('-rd','--root_dir', type=str, help='training files are refered  from this folder', default='./sample_data')
parser.add_argument('--crop', action='store_true', help='use crop', default=False)
parser.add_argument('--rotate', action='store_true', help='whether to augment by roatating the image 90/180/270', default=False)
parser.add_argument("-gf", "--gamma_factor", type=float, default=1.0)
parser.add_argument('--random_focal_length', action='store_true', help='whether to generate training images with random focal length', default=False)
parser.add_argument('--augment_illumination', action='store_true', help='whether to generate images with augmented illumination', default=False)
parser.add_argument('-ji','--jitter_illumination', type=float, help='jitter_illumination', default=1.0)
parser.add_argument('-rflo','--random_focal_length_octaves', type=float, help='random focal length octaves', default=3.0)
parser.add_argument('-rfl','--reg_factor_length', type=float, help='regularization factor length', default=1.0)
parser.add_argument('--augment_trajectories', action='store_true', help='whether to augment the trajectoires', default=False)
parser.add_argument("-nl", "--noise_level", type=float, default=0.01)

args = parser.parse_args()

# Hyper Parameters
IMAGE_SIZE = args.imagesize
OUTPUT_DIR = args.output_dir
N_POSITIONS = args.n_positions  # number of filters
GPU=0

ROOT_DIR = args.root_dir
GAMMA_FACTOR = args.gamma_factor



def save_tensor_img(tensor, filename):
    img = tensor[0].permute(1,2,0).detach().cpu().numpy() + 0.5
    imsave(filename, (255*np.clip(img,0,1)).astype(np.uint8))

def main():


    try:
        best_result = 1000000
        print("cuda is available: ", torch.cuda.is_available())
        
        if os.path.exists(OUTPUT_DIR) == False:
            os.system('mkdir -p ' + OUTPUT_DIR )

        with open(os.path.join(OUTPUT_DIR, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        
        train_dataset = BlurrySharpPairDatasetOnline(args.sharp_folder, args.positions_folder, 
                                                      n_positions=N_POSITIONS, rotate=args.rotate, 
                                                      random_focal_length=args.random_focal_length,
                                                      random_focal_length_octaves=args.random_focal_length_octaves, 
                                                      augment_illumination=args.augment_illumination, 
                                                      jitter_illumination=args.jitter_illumination,
                                                      augment_trajectories=args.augment_trajectories)
        num_samples=20                                                  
        rand_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=num_samples)  #40000
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, sampler=rand_sampler)  # shuffle=True

        print('data loader length = %d' % len(train_dataloader))
        number_of_steps_per_epoch = len(train_dataloader)

       
        for epoch in range(1):
            
            for iteration, images in enumerate(train_dataloader):

                gt_sharp = images['sharp_image'].cuda(GPU)
                intrinsics_gt=images['intrinsics'].cuda(GPU)
                camera_positions_gt = images['positions'].cuda(GPU) # BxNx6

                with torch.no_grad():
                    offsets = get_offsets_from_positions(gt_sharp.shape,camera_positions_gt, intrinsics_gt)
                    blurry = reblur_offsets(gt_sharp, offsets)
                    #blurry, _ = reblur_homographies_v3(gt_sharp, camera_positions_gt, intrinsics_gt)
                    blurry = torch.clamp(blurry,0,1)
                #print(blurry.shape, gt_sharp.shape)

                B,C,H,W = blurry.shape
                print(blurry.shape)
                print(blurry.min(),blurry.max())

                blurry_image_gamma_inverted = blurry  ** (1.0 / GAMMA_FACTOR)
                gt_image_gamma_inverted = gt_sharp  ** (1.0 / GAMMA_FACTOR)
                save_tensor_img(blurry_image_gamma_inverted - 0.5, os.path.join(OUTPUT_DIR, 'iter_%d_img_blurry.png' % iteration))
                save_tensor_img(gt_image_gamma_inverted - 0.5, os.path.join(OUTPUT_DIR, 'iter_%d_img_gt.png' % iteration))    
                save_kernels_from_offsets(offsets[0],os.path.join(OUTPUT_DIR, 'iter_%d_kernels.png' % iteration))

                print('Training images saved')
                ############################################################################
                ################       END  IMAGES LOG             #########################
                ############################################################################


        return

    except Exception as e:
        torch.cuda.empty_cache()
        print(e)
        traceback.print_exc()
        return

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
