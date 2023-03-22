import os
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets  
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from models import Generator, Discriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)





#caracter deterministic
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print("Random seed:", manualSeed)

#inputs
dataroot = r"D:\faia\cod\detectaremasca\Data\asd"
batch_size = 64
image_size = 64

nz = 100

nc = 3  
ngf = 64    #pentru numarul de filtre din generator ca sa fie in mod dinamic
ndf = 64    #numarul de filtre(feature maps) din discriminator

num_epochs = 500
lr = 0.0002
beta1 = 0.5 #pt otpimizatorul adam


#dataset
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size), #pt aspect ratio
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #standardizare + normalizare 0,1 
])

#dataset
dataset = torchvision.datasets.ImageFolder(root = dataroot, transform = transforms)

#dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#functie pentru initializare custom a weighturilor apelata la netG si netD

#instantiere 
netG = Generator(nz,ngf, nc).to(device)
netG.apply(weights_init)

netD = Discriminator(ndf, nc).to(device)
netD.apply(weights_init)

#loss
criterion = nn.BCELoss()

#zgomot random fix (100,1,1)
fixed_noise = torch.rand(batch_size, nz, 1, 1, device = device)

#optimizer
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))


iters = 0
img_list = []
G_losses = []
D_losses = []

#training loop
for epoch in range(num_epochs):
    print(epoch)
    for i, data in enumerate(dataloader):

        #antrenare discriminator
        netD.zero_grad()
        real_images = data[0].to(device)

        # 64 elem -> [1,1,1,1...]
        #ii arat imagini reale si ii spun ca sunt reale
        label_true = torch.full((batch_size, ), 1, dtype = torch.float, device = device)
        output = netD(real_images).view(-1)  # (64, 1) -> (64, )
        errD_real = criterion(output, label_true)
        errD_real.backward()

        #ii arata imagini generate si ii spun ca sunt generate
        noise = torch.rand(batch_size, nz, 1, 1, device=device) # (64, 100, 1, 1)
        fake = netG(noise)
        label_false = torch.full((batch_size, ), 0, dtype = torch.float, device = device)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label_false)
        errD_fake.backward()

        optimizerD.step()


        #antrenare generator
        netG.zero_grad()
        label_true = torch.full((batch_size, ), 1, dtype = torch.float, device = device)
        output = netD(fake).view(-1)
        errG = criterion(output, label_true)
        errG.backward()
        optimizerG.step()


        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            if not os.path.exists("generated"):
                os.mkdir("generated")

            img_to_save = np.transpose(img_list[-1].numpy(), (1,2,0))
            plt.imsave(os.path.join('generated', f"epoch{epoch}_iter{iters}.png"), img_to_save)
        iters += 1


    torch.save(netG.state_dict(), os.path.join("generated", f"epoch{epoch}"))



