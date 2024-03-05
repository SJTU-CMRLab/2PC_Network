# import libraries
import os

import time
import logging
import random
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_data, TraindataSet, randomTransform
from network import Net
from LossVGG import VGGLoss

# -------------------------------------#
# ---------Initialization--------------#
# -------------------------------------#

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set GPU device
gpu_list = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set directories
# data directory
inputSax_dir = "./data/Train/Input"
labelSax_dir = "./data/Train/Label"
# model directory
model_dir = './model'
os.makedirs(model_dir, exist_ok=True)
# log directory
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

# create a logger
logger = logging.getLogger('mylogger')
# set logger level
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(log_dir + '/mylog.txt')
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# tensorboard
writer = SummaryWriter("logger")

# -------------------------------------#
# -------------Load data---------------#
# -------------------------------------#
# load data
train_data, train_label = get_data(inputSax_dir, labelSax_dir)

# -------------------------------------#
# -------------Define paras------------#
# -------------------------------------#

# define loss function
loss_MSE = nn.MSELoss()
loss_VGG = VGGLoss()

# define the number of epochs
start_epoch = 0
end_epoch = 400
# learning rate
lr = 1e-4
# batch size
batch_size = 1

# -------------------------------------#
# -------------Define network----------#
# -------------------------------------#

# Dual-encoder neural network
net2PC = Net()
net2PC = net2PC.to(device)
if start_epoch > 0:
    net2PC.load_state_dict(torch.load('%s/net2PC_params_%d.pkl' % (model_dir, start_epoch)))

# optimizer
optimizer = torch.optim.Adam(params=net2PC.parameters(), lr=lr)

# -------------------------------------#
# -------------Start training----------#
# -------------------------------------#

start = time.time()
numIter = 0  # to record the number of iterations
for epoch in range(start_epoch + 1, end_epoch + 1):

    # Training process
    dataset = TraindataSet(train_data, train_label, transform=randomTransform)
    train_iter = DataLoader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True)

    net2PC.train()
    i = 0  # to record the number of batch
    for x, y in train_iter:

        i = i + 1
        numIter = numIter + 1
        torch.cuda.empty_cache()

        label = y.to(device, non_blocking=True)
        input = x.to(device, non_blocking=True)

        # dual-encoder neural network
        output1 = net2PC(input)
        input_reverse = torch.cat((torch.unsqueeze(input[:, 1, :, :, :], 1), torch.unsqueeze(input[:, 0, :, :, :], 1)), 1)
        output2 = net2PC(input_reverse)
        output = (output1 + output2) / 2

        # loss
        train_loss_MSE = loss_MSE(output, label)
        train_loss_VGG = loss_VGG(output, label)
        loss = 1 * train_loss_MSE + 0.01 * train_loss_VGG

        # update AS network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss
        print(
            "[Epoch %d/%d] [Batch %d/%d] [MSE: %f] [VGG: %f]"
            % (epoch, end_epoch, i, len(train_iter), train_loss_MSE.item(), train_loss_VGG.item()))
        logger.info("[Epoch %d/%d] [Batch %d/%d] [MSE: %f] [VGG: %f]"
                    % (epoch, end_epoch, i, len(train_iter), train_loss_MSE.item(), train_loss_VGG.item()))

        writer.add_scalars('Loss/TrainMSE', {'MSE': train_loss_MSE.item()}, numIter)
        writer.add_scalars('Loss/TrainVGG', {'VGG': train_loss_VGG.item()}, numIter)

        # to record images
        input = torch.reshape(torch.squeeze(input), [50, 1, 192, 192])
        label = torch.unsqueeze(torch.squeeze(label), 1)
        output1 = torch.unsqueeze(torch.squeeze(output1), 1)
        output2 = torch.unsqueeze(torch.squeeze(output2), 1)
        output = torch.unsqueeze(torch.squeeze(output), 1)
        if numIter % 125 == 1:
            writer.add_images('TrainImages/Input', input, int((numIter - 1) / 125), dataformats='NCHW')
            writer.add_images('TrainImages/Output1', output1, int((numIter - 1) / 125), dataformats='NCHW')
            writer.add_images('TrainImages/Output2', output2, int((numIter - 1) / 125), dataformats='NCHW')
            writer.add_images('TrainImages/Output', output, int((numIter - 1) / 125), dataformats='NCHW')
            writer.add_images('TrainImages/Label', label, int((numIter - 1) / 125), dataformats='NCHW')


    # save AS models
    if epoch % 5 == 0:
        torch.save(net2PC.state_dict(), "%s/net2PC_params_%d.pkl" % (model_dir, epoch))

# time summary
train_duration = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(train_duration // 60, train_duration % 60))
logger.info('Training complete in {:.0f}m {:.0f}s'.format(train_duration // 60, train_duration % 60))

writer.close()
