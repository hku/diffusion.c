import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import random
import torch.nn.init as init
import struct
from dataclasses import dataclass, fields
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

SIZE_K = 3 #kernel_size in conv
SIZE_P = 1 #padding_size in conv

@dataclass
class Config:
    width: int = 16
    height: int = 16
    n_in: int = 3 # in_channels, dfault to 3
    n_feature: int = 64 # out_chanels
    n_cfeature: int = 5 #contxt embeding size


cfg = Config(**dict(
    n_feature = 64,
))

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def serialize(t: torch.Tensor):
    d = t.detach().cpu().view(-1).numpy().astype(np.float32)
    b = struct.pack(f'{len(d)}f', *d)
    return b

class ExportMixin:
    def export (self, f):
        for layer in self.layers_export:
            if hasattr(layer, 'export'):
                layer.export(f)
            elif isinstance(layer, nn.Module):
                if hasattr(layer, 'weight'):
                    f.write(serialize(layer.weight))
                if hasattr(layer, 'bias'):
                      f.write(serialize(layer.bias))


class ResidualConvBlock(nn.Module):
    def __init__(self, n_in, n_out, is_res=False):
        super().__init__()
        self.is_res = is_res
        self.same_channels = n_in == n_out

        size_k = SIZE_K
        size_p = SIZE_P

        conv1_core = nn.Conv2d(n_in, n_out, size_k, 1, size_p)
        conv2_core = nn.Conv2d(n_out, n_out, size_k, 1, size_p)
        conv3_core = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0) if is_res and (n_in != n_out) else None


        self.conv1_core = conv1_core
        self.conv2_core = conv2_core
        self.conv3_core = conv3_core

        self.conv1 = nn.Sequential(
            conv1_core,  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(n_out),  # Batch normalization
            GELU(),  # GELU activation function
            # nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            conv2_core,  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(n_out),  # Batch normalization
            GELU(),  # GELU activation functions
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        if self.is_res:
            if self.same_channels:
                out = x + x2
            else:
                shortcut = self.conv3_core(x)
                out = shortcut + x2
            return out / 1.414
        else:
            return x2
    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

    def export(self, f):
        f.write(serialize(self.conv1_core.weight))
        f.write(serialize(self.conv1_core.bias))
        f.write(serialize(self.conv2_core.weight))
        f.write(serialize(self.conv2_core.bias))
        if self.conv3_core is not None:
            f.write(serialize(self.conv3_core.weight))
            f.write(serialize(self.conv3_core.bias))


class UnetDown(nn.Module, ExportMixin):
    def __init__(self, n_in, n_out):
        super().__init__()

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        block1 = ResidualConvBlock(n_in, n_out)
        block2 = ResidualConvBlock(n_out, n_out)

        self.layers_export = [block1, block2]

        # layers = [block1, block2]
        layers = [block1, block2, nn.MaxPool2d(2)]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)


class UnetUp(nn.Module, ExportMixin):
    def __init__(self, n_in, n_out):
        super().__init__()


        block0 = nn.ConvTranspose2d(n_in, n_out, 2, 2)
        block1 = ResidualConvBlock(n_out, n_out)
        block2 = ResidualConvBlock(n_out, n_out)

        self.layers_export = [block0, block1, block2]

        layers = self.layers_export

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module, ExportMixin):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        block1 = nn.Linear(n_in, n_out)
        block2 = nn.Linear(n_out, n_out)

        self.layers_export = [block1, block2]

        layers = [block1, GELU(), block2]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.n_in)
        return self.model(x)


class Out(nn.Module, ExportMixin):
    def __init__(self, n_in, n_out):
        super().__init__()

        size_k = SIZE_K
        size_p = SIZE_P

        conv1 = nn.Conv2d(2 * n_out, n_out, size_k, 1, size_p)
        conv2 = nn.Conv2d(n_out, n_in, size_k, 1, size_p)

        self.layers_export = [conv1, conv2]

        layers = [conv1, nn.GroupNorm(8, n_out), nn.ReLU(), conv2]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class ContextUnet(nn.Module, ExportMixin):
    def __init__(self, n_in, n_feat, n_cfeat=5, image_size=16):
        super().__init__()

        self.n_in = n_in
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.image_size = image_size

        block_in = ResidualConvBlock(n_in, n_feat, True)
        down1 = UnetDown(n_feat, n_feat)
        down2 = UnetDown(n_feat, 2 * n_feat)
        to_vec = nn.Sequential(nn.AvgPool2d((4)), GELU())

        timeembed1 = EmbedFC(1, 2 * n_feat)
        timeembed2 = EmbedFC(1, n_feat)
        contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        contextembed2 = EmbedFC(n_cfeat, n_feat)

        up0_size_k = image_size //4
        up0_size_s = image_size //4

        up0_core = nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, up0_size_k, up0_size_s)
        up0 = nn.Sequential(up0_core, nn.GroupNorm(8, 2 * n_feat), nn.ReLU())

        up1 = UnetUp(4 * n_feat, n_feat)
        up2 = UnetUp(2 * n_feat, n_feat)

        block_out = Out(n_in, n_feat)

        self.layers_export = [
            block_in,
            down1, down2,
            timeembed1, timeembed2, contextembed1, contextembed2,
            up0_core, up1, up2,
            block_out
        ]

        self.block_in = block_in
        self.down1 = down1
        self.down2 = down2
        self.to_vec = to_vec
        self.timeembed1 = timeembed1
        self.timeembed2 = timeembed2
        self.contextembed1 = contextembed1
        self.contextembed2 = contextembed2
        self.up0 = up0
        self.up1 = up1
        self.up2 = up2
        self.block_out = block_out


    def forward(self, x, t, c):
        x = self.block_in(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        cemb1 = self.contextembed1(c).view(-1, cfg.n_feature * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, cfg.n_feature * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, cfg.n_feature, 1, 1)
        temb2 = self.timeembed2(t).view(-1, cfg.n_feature, 1, 1)

        up0 = self.up0(hiddenvec)
        up0 = cemb1 * up0 + temb1

        up1 = self.up1(up0, down2)
        up1 = cemb2 * up1 + temb2

        up2 = self.up2(up1, down1)

        out = self.block_out(torch.cat((up2, x), 1))

        return out


class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape

transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

timesteps = 500
beta1 = 1e-4
beta2 = 0.02

def train():
    print("training...")
    save_dir = os.path.abspath('./weights')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

    nn_model = ContextUnet(cfg.n_in, n_feat=cfg.n_feature, n_cfeat=cfg.n_cfeature, image_size=cfg.height).to(device)

    # training hyperparameters
    batch_size = 100
    n_epoch = 32
    lrate = 1e-3

    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    # def2.2
    a_t = 1 - b_t
    # def2.3
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1


    # load dataset and construct optimizer
    dataset = CustomDataset("./data/sprites_1788_16x16.npy", "./data/sprite_labels_nc_1788_16x16.npy", transform,
                            null_context=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    nn_model.train()

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader, mininterval=2 )
        for x, c in pbar:   # x: images
            optim.zero_grad()

            x = x.to(device)
            c = c.to(torch.float32).to(device)



            # perturb data
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)


            # 公式2.5
            x_pert = perturb_input(x, t, noise)

            # use network to recover noise
            pred_noise = nn_model(x_pert, t / timesteps, c)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()

            optim.step()

        # save model periodically
        if ep%4==0 or ep == int(n_epoch-1):
            torch.save(nn_model.state_dict(), os.path.join(save_dir, f"ckpt_{ep}.pth"))
            print(f"saved model ckpt_{ep}.pth")

    # print("loading weights")
    # nn_model.load_state_dict(torch.load(f"weights/context_model_31.pth", map_location=device))

    # print("check point keys:")
    # checkpoint = torch.load(f"weights/context_model_31.pth", map_location=device)
    # ckeys = checkpoint.keys()
    # # ckeys = [k for k in ckeys if k.startswith("init_conv")]
    # print(ckeys)
    # print(len(ckeys))
    #
    # print("==============================")
    # print("model keys:")
    # mkeys = nn_model.state_dict().keys()
    # mkeys = [k for k in mkeys if k.startswith("block_in")]
    #
    # print(mkeys)
    # print(len(mkeys))


def infer(timesteps=500, beta2=0.02, beta1=1e-4):

    save_dir = os.path.abspath('./weights')
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    nn_model = ContextUnet(cfg.n_in, n_feat=cfg.n_feature, n_cfeat=cfg.n_cfeature, image_size=cfg.height).to(device)

    nn_model.load_state_dict(torch.load(f"{save_dir}/ckpt_31.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Model")


    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    # def2.2
    a_t = 1 - b_t
    # def2.3
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1



    def denoise_add_noise(x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = b_t.sqrt()[t] * z

        # 公式2.13d
        mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        # 公式2.16
        return mean + noise

    @torch.no_grad()
    def sample_ddpm(n_sample, save_rate=20):


        # x_T ~ N(0, 1), sample initial noise

        samples = torch.randn(n_sample, 3, cfg.height, cfg.width).to(device)

        # array to keep track of generated steps for plotting
        intermediate = []
        for i in range(timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

            c = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]).view(-1, cfg.n_cfeature).to(device)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            ## nn_model 即描述 epsilon_\theta 的神经网络
            eps = nn_model(samples, t, c)  # predict noise e_(x_t,t)

            ## 公式2.16
            samples = denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        return samples, intermediate

    samples, intermediate_ddpm = sample_ddpm(4)

    return samples, intermediate_ddpm


def export():
    print("exporting bin file for c...")
    save_dir = os.path.abspath('./weights')
    binpath = os.path.join(save_dir, 'ckpt.bin')

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    nn_model = ContextUnet(cfg.n_in, n_feat=cfg.n_feature, n_cfeat=cfg.n_cfeature, image_size=cfg.height).to(device)
    nn_model.load_state_dict(torch.load(f"{save_dir}/ckpt_31.pth", map_location=device))
    nn_model.eval()

    with open(binpath, 'wb') as f:
        for field in fields(Config):
            b = struct.pack('i', getattr(cfg, field.name))
            f.write(b)
        nn_model.export(f)
        print(f"wrote to {binpath}")
    pass

if __name__ == '__main__':
    train()
    export()