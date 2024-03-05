# import libraries
import os

import time
import random
import torch
import numpy as np

from utils import normalization
from network import Net
from scipy.io import loadmat
import scipy.io as sio

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
# model directory
model_dir = './model'

# -------------------------------------#
# ----------Testing process------------#
# -------------------------------------#

# choose model
epoch_num = 400
net2PC = Net()
net2PC = net2PC.to(device)
net2PC.load_state_dict(torch.load('%s/net2PC_params_%d.pkl' % (model_dir, epoch_num)))
net2PC.eval()

# output directory
outputDir = './TestOutput'
os.makedirs(outputDir, exist_ok=True)

# testing kernel
def TestKernel(input):

    with torch.no_grad():
        start = time.time()

        input = input.cuda()
        output1 = net2PC(input)

        input_reverse = torch.cat(
            (torch.unsqueeze(input[:, 1, :, :, :], 1), torch.unsqueeze(input[:, 0, :, :, :], 1)), 1)
        output2 = net2PC(input_reverse)

        output = (output1 + output2) / 2


        duration = time.time() - start
        print('Testing complete in {:.0f}m {:.2f}s'.format(duration // 60, duration % 60))

        return output


# load testing data
dataDir = './data/Test'
crrDataName = dataDir + "/CropSax" + str(1) + ".mat"
crrData = loadmat(crrDataName)
crrData = torch.FloatTensor(crrData['CropSax'])
crrData = torch.unsqueeze(crrData, 0)
crrData = normalization(crrData)

output = TestKernel(input=crrData)

arrayOutput = np.array(torch.squeeze(output).cpu()).astype(np.float64)
matName = "output"
mdic = {matName: arrayOutput}
sio.savemat(os.path.join(outputDir, matName + '.mat'), mdic)



