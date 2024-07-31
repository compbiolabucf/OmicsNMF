import torch, torch.nn as nn, os
from codes.models import Generator, Critic
import torch.optim as optim
import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation
from codes.utils import nmf_U_V,get_prediction_auc
from collections import Counter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
from codes.dataloader import get_dataloaders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-source", type=str)
parser.add_argument("-target", type=str)
parser.add_argument("-label",default=None,type=str)
parser.add_argument("-key",default="omics_nmf_train")
parser.add_argument("-save_dir",default="saved_models")
parser.add_argument("-bs",default=64,type=int,help="Batch size")
parser.add_argument("-k",default=10,type=int,help="cluster size")
parser.add_argument("-epochs",default=2,type=int,help="epochs")
parser.add_argument("-clip", default = 0.01, type=float, help = "wgan cliping limit")
parser.add_argument("-n_critic", default = 5, type=int, help = "critic train iteration")
parser.add_argument("-lr",default=0.00005,type=float,help="learning rate")
parser.add_argument("-beta1",default=0.5,type=float,help="beta1 for optim")
parser.add_argument("-U",default=0.1,type=float,help="nmf loss multiplier")
parser.add_argument("-M",default=0.1,type=float,help="mse loss multiplier")
parser.add_argument("-gen",default=1.0,type=float,help="gen loss multiplier")
parser.add_argument("-d", default=0, type=int, help="Gpu index")

args = parser.parse_args()

source_path = args.source
target_path = args.target
label_path = args.label
if(label_path is None):
    print("No label information is provided!! Model will be saved based on lowest mean square error instead of AUC.")
else:
    print("Label information is provided!! Model will be saved based on best AUC on classfication task.")
k = args.k # number of cluster for nnmf
key = args.key
batch_size = args.bs
lr = args.lr
beta1 = args.beta1
clip = args.clip
n_critic = args.n_critic
num_epochs = args.epochs
device = f'cuda:{args.d}'
mse_loss_multiplier = args.M
U_loss_multiplier = args.U
gen_loss_multiplier = args.gen
save_dir = args.save_dir


train_loader, valid_loader, test_loader, source_feat, target_feat = get_dataloaders(source_path, target_path, batch_size=batch_size)

real_label = 1.0
fake_label = 0.0

netG = Generator(input_size = source_feat, output_size = target_feat)
netD = Critic(input_size = target_feat)
print(netG)
print(netD)
netG.to(device)
netD.to(device)

optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

criterion_bce = nn.BCELoss()
criterion_mse = nn.MSELoss()

save_path = f"{save_dir}/{key}/"
os.makedirs(save_path, exist_ok=True)
writer = SummaryWriter(log_dir = save_path)

"""for the Y1 (available microRNA data generate U ahead of training)"""
targetomics = train_loader.dataset.Y1.values
U_Global, V_Global = nmf_U_V(targetomics,k=k)


best_auc = 0
best_epoch = 0
best_mse = 99999999999
for epoch in range(num_epochs):
    G_losses = []
    D_losses = []
    U_losses = []
    G_U_losses = []
    MSE_losses = []
    for i,data in enumerate(tqdm(train_loader)):
        netD.zero_grad()
        x,s,xs,y,label = data
        label_placeholder = torch.full((xs.size(0),), real_label, dtype = torch.float, device = device)
        x = x.to(device)
        y = y.to(device)
        label = label.to(device)

        n_critic_loss = []
        for _ in range(n_critic):
            output_real = netD(y).view(-1)
            output_fake = netD(netG(x)).view(-1)

            loss_critic_real = torch.mean(output_real)
            loss_critic_fake = torch.mean(output_fake)

            loss_critic = - (loss_critic_real-loss_critic_fake)
            n_critic_loss.append(loss_critic.item())
            netD.zero_grad()
            loss_critic.backward()
            optimizerD.step()
            
            for p in netD.parameters():
                p.data.clamp_(-clip, clip)
        D_losses.append(np.mean(n_critic_loss))


        # noise = xs
        netG.zero_grad()
        fake = netG(x)
        output = netD(fake).view(-1)
        loss_mse = criterion_mse(y, fake)
        U_hat,V_hat = nmf_U_V(fake.T.detach().cpu())
        loss_U = criterion_mse(torch.tensor(U_Global), torch.tensor(U_hat))
        loss_Gz = -torch.mean(output) + loss_U*U_loss_multiplier + loss_mse*mse_loss_multiplier
        netG.zero_grad()
        loss_Gz.backward()
        optimizerG.step()

        G_losses.append(loss_Gz.item())
        # G_U_losses.append(loss_G.item())
        MSE_losses.append(loss_mse.item())
        U_losses.append(loss_U.item())
        # Output training stats
    
    for i,data in enumerate(tqdm(test_loader)):
        netD.zero_grad()
        x,s,xs,y,label = data
        label_placeholder = torch.full((xs.size(0),), real_label, dtype = torch.float, device = device)
        x = x.to(device)
        y = y.to(device)
        label = label.to(device)
        # noise = xs
        netG.zero_grad()
        fake = netG(x)
        output = netD(fake).view(-1)
        U_hat,V_hat = nmf_U_V(fake.T.detach().cpu())
        loss_U = criterion_mse(torch.tensor(U_Global), torch.tensor(U_hat))
        loss_Gz = -torch.mean(output) + loss_U*U_loss_multiplier
        netG.zero_grad()
        loss_Gz.backward()
        optimizerG.step()


    writer.add_scalar("Loss/Discriminator", np.mean(D_losses), epoch)
    writer.add_scalar("Loss/Generator", np.mean(G_losses), epoch)
    writer.add_scalar("Loss/MSE", np.mean(MSE_losses), epoch)
    writer.add_scalar("Loss/Gen+U", np.mean(G_U_losses), epoch)
    writer.add_scalar("Loss/U_loss", np.mean(U_losses), epoch)

    netG.eval()
    valid_LOSSES_U = []
    valid_LOSSES_MSE = []

    
    for i,data in enumerate(tqdm(valid_loader)):
        x,s,xs,y,label = data
        x = x.to(device)
        y = y.to(device)

        y_hat = netG(x)
        loss = criterion_mse(y, y_hat)
        
        U_hat,V_hat = nmf_U_V(y_hat.T.detach().cpu())
        valid_loss_U = criterion_mse(torch.tensor(U_Global), torch.tensor(U_hat))

        valid_LOSSES_U.append(valid_loss_U.item())
        valid_LOSSES_MSE.append(loss.item())
    mse_loss = np.mean(valid_LOSSES_MSE)
    mean_auc, auc_test, auc_valid, auc_plot, target_df = get_prediction_auc(train_loader, valid_loader, test_loader, netG, label_path, device = device)
    
    if(label_path is None):
        if(best_mse>mse_loss):
            best_mse = mse_loss
            best_epoch = epoch
            torch.save({
                'epoch':epoch, 
                'mse':mse_loss,
                'gen_state_dict':netG.state_dict(),
                'dis_state_dict':netD.state_dict(),
                'optimizerD':optimizerD.state_dict(),
                'optimizerG':optimizerG.state_dict(),
                'auc':mean_auc,
            },  save_path + f"model_dict.pth")
            target_df.to_csv(save_path+ "imputed_data.csv", index=None)
    else:
        if(mean_auc>best_auc):
            best_auc = mean_auc
            best_epoch = epoch
            torch.save({
                'epoch':epoch, 
                'mse':mse_loss,
                'gen_state_dict':netG.state_dict(),
                'dis_state_dict':netD.state_dict(),
                'optimizerD':optimizerD.state_dict(),
                'optimizerG':optimizerG.state_dict(),
                'auc':mean_auc,
            },  save_path + f"model_dict.pth")
            target_df.to_csv(save_path+ "imputed_data.csv", index=None)

    writer.add_scalar("Loss/Validation_MSE", np.mean(valid_LOSSES_MSE), epoch)
    writer.add_scalar("Loss/Validation_U", np.mean(valid_LOSSES_U), epoch)
    writer.add_scalar("Loss/mean_auc", mean_auc, epoch)
    writer.add_scalar("Loss/tested_on_missing", auc_test, epoch)
    writer.add_scalar("Loss/tested_on_valid", auc_valid, epoch)

print("Finished Training")
if(label_path is None):
    print("Best MSE: ", best_mse, " on epoch ", best_epoch)
else:
    print("Best AUC: ", best_auc, " on epoch ", best_epoch)
    print("Auc on validation set: ", auc_valid)
    print("Auc on test set: ", auc_test)