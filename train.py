import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from models import *
from datasets import *
import numpy as np
import nibabel as nib
import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=27)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument("--sample_interval", type=int, default=140, help="interval between image sampling")
parser.add_argument("--out_path", default="../data/recon/")
parser.add_argument("--semi", default="L")

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
lambda_gp = 10


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


adversarial_loss = torch.nn.BCELoss()
voxelwise_loss = torch.nn.L1Loss()

generator = Generator()
discriminator = Discriminator()
print("# parameters:", sum(param.numel() for param in generator.parameters()))

if torch.cuda.device_count() > 1:
    print("let's use ", torch.cuda.device_count(), "GPUs")
    generator = torch.nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
    discriminator = torch.nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))

if cuda:
    adversarial_loss.cuda()
    voxelwise_loss.cuda()
    generator.cuda()
    discriminator.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if opt.netG != '':
    generator.load_state_dict(torch.load(opt.netG)['state_dict'])
    print("continue train netG")


if opt.netD != '':
    discriminator.load_state_dict(torch.load(opt.netD)['state_dict'])
    print("continue train netD")



data_train = DataLoader(ImageDatasets("../data/train/"),
                        batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.n_cpu, drop_last=True)
data_test = DataLoader(ImageDatasets("../data/val/"), batch_size=70,
                       shuffle=False, num_workers=1)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

AFFINE = np.array([[-1.5, 0., 0., 90.],
                   [0., 1.5, 0., -126.],
                   [0., 0., 1.5, -72.],
                   [0., 0., 0., 1.]])


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake1 = Tensor(real_samples.shape[0], 1).fill_(1.0)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake1,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def save_sample(ite):
    generator.eval()
    with torch.no_grad():
        l_img, r_img, raw_img = next(iter(data_test))
        l_img = l_img.type(Tensor)
        r_img = r_img.type(Tensor)
        generator.eval()
        if opt.semi == 'L':
            gen_mask = generator(l_img)
            err = voxelwise_loss(gen_mask, r_img)
        else:
            gen_mask = generator(r_img)
            err = voxelwise_loss(gen_mask, l_img)
    print('test_MSEloss:', err.item())
    if opt.semi == 'L':
        raw_img[:, :, 60:115, 15:134, 15:102] = gen_mask
    else:
        raw_img[:, :, 5:60, 15:134, 15:102] = gen_mask
    raw_img = raw_img.cpu()
    raw_img = raw_img.detach()
    for index in np.arange(3):
        recon = raw_img[index, :, :, :, :].numpy()
        recon = np.squeeze(recon)
        img = nib.Nifti1Image(recon, affine=AFFINE)
        nib.save(img, "../recon/%d_%d.nii" % (ite, index))


for epoch in range(opt.n_epochs):
    for i, (l_img, r_img, raw_img) in enumerate(data_train):
        generator.train()
        discriminator.train()
        # valid = Tensor(masked_img.shape[0], 1).fill_(1.0)
        # fake = Tensor(masked_img.shape[0], 1).fill_(0.0)
        l_img = l_img.type(Tensor)
        r_img = r_img.type(Tensor)
        if opt.semi == 'L':
            masked_data = r_img
            masked_part = l_img
        else:
            masked_data = l_img
            masked_part = r_img

        optimizer_G.zero_grad()
        gen_part = generator(masked_part)

        # g_adv = adversarial_loss(discriminator(gen_part), valid)
        g_adv = -torch.mean(discriminator(gen_part))
        g_voxel = voxelwise_loss(gen_part, masked_data)
        g_loss = 0.002 * g_adv + 0.998 * g_voxel

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        # real_loss = adversarial_loss(discriminator(masked_part), valid)
        # fake_loss = adversarial_loss(discriminator(gen_part.detach()), fake)
        gradient_penalty = compute_gradient_penalty(discriminator, masked_data.data, gen_part.data)
        # d_loss = 0.5 * (real_loss + fake_loss) + lambda_gp * gradient_penalty
        d_loss = -torch.mean(discriminator(masked_data)) + torch.mean(
            discriminator(gen_part.detach())) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [GP: %f] [G adv: %f, pixel: %f]"
            % (epoch + 1, opt.n_epochs, i + 1, len(data_train), d_loss.item(), gradient_penalty.item(), g_adv.item(),
               g_voxel.item())
        )

        batches_done = epoch * len(data_train) + i + 1
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

if opt.semi == 'L':
    out_pathG = ''
    out_pathD = ''

else:
    out_pathG = ''
    out_pathD = ''

torch.save({'epoch': epoch + 1,
            'state_dict': generator.state_dict()},
           out_pathG)

torch.save({'epoch': epoch + 1,
            'state_dict': discriminator.state_dict()},
           out_pathD)
