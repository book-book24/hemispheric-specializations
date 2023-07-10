import ants
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.io import savemat, loadmat

tmp_dir = '../data/tmp/'
path_raw = '../data/test/'
path_recon_lr = '../data/recon_lr/'
path_recon_rl = '../data/recon_rl/'
path_J_lr = '../data/J_lr/'
path_J_rl = '../data/J_rl/'
images = [os.path.join(path_raw, img) for img in os.listdir(path_raw)]
N = len(images)

for i in range(N):
    raw_nii = ants.image_read(images[i])
    raw_arr = raw_nii.numpy()
    raw_image_lr = np.zeros(raw_nii.shape)
    raw_image_rl = np.zeros(raw_nii.shape)
    lr_image = np.zeros(raw_nii.shape)
    rl_image = np.zeros(raw_nii.shape)
    recon_name = os.path.basename(os.path.splitext(images[i])[0] + '.mat')
    recon_lr = loadmat(os.path.join(path_recon_lr, recon_name))['recon'].squeeze()
    recon_rl = loadmat(os.path.join(path_recon_rl, recon_name))['recon'].squeeze()
    raw_image_lr[60:115, 15:134, 15:102] = raw_arr[60:115, 15:134, 15:102]
    raw_image_rl[5:60, 15:134, 15:102] = raw_arr[5:60, 15:134, 15:102]
    lr_image[60:115, 15:134, 15:102] = recon_lr
    rl_image[5:60, 15:134, 15:102] = recon_rl

    raw_image_lr = raw_nii.new_image_like(raw_image_lr)
    raw_image_rl = raw_nii.new_image_like(raw_image_rl)
    lr_image = raw_nii.new_image_like(lr_image)
    rl_image = raw_nii.new_image_like(rl_image)

    tx_lr = ants.registration(fixed=raw_image_lr, moving=lr_image,
                              type_of_transform='SyN', outprefix=tmp_dir)

    J_lr = ants.create_jacobian_determinant_image(domain_image=lr_image, tx=tx_lr['fwdtransforms'][0])
    J_lr = J_lr - 1

    J_lr.to_filename(os.path.join(path_J_lr, os.path.basename(images[i])))

    tx_rl = ants.registration(fixed=raw_image_rl, moving=rl_image,
                              type_of_transform='SyN', outprefix=tmp_dir)

    J_rl = ants.create_jacobian_determinant_image(domain_image=rl_image, tx=tx_rl['fwdtransforms'][0])
    J_rl = J_rl - 1

    J_rl.to_filename(os.path.join(path_J_rl, os.path.basename(images[i])))
