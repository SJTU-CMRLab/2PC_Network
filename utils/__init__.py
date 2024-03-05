import torch
import random
import numpy as np
import cv2 as cv

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset


# Load data
def get_data(input_dir, label_dir):

    data_train, label_train = None, None

    volunteerTrain_list = [1]

    # 180 interval
    for m in range(len(volunteerTrain_list)):
        voltIdx = volunteerTrain_list[m]
        for n in range(6):
            freqTrain_list = [n + 1, n + 7]
            x_train, y_train = None, None

            # load 2PC data
            for j in range(len(freqTrain_list)):
                freqIdx = freqTrain_list[j]

                crrData_dir = input_dir + '/' + "RegisSax" + str(voltIdx) + "-" + str(freqIdx) + ".mat"
                crrData = loadmat(crrData_dir)
                crrData = torch.FloatTensor(crrData['RegisSax'])
                crrData = normalization(torch.unsqueeze(crrData, 1))

                crrLabel_dir = label_dir + '/' + "AveLabel" + str(voltIdx) + ".mat"
                crrLabel = loadmat(crrLabel_dir)
                crrLabel = torch.FloatTensor(crrLabel['AveLabel'])
                crrLabel = normalization(torch.unsqueeze(crrLabel, 1))

                if x_train is None:
                    x_train = crrData
                    y_train = crrLabel

                else:

                    x_train = torch.cat((x_train, crrData), 1)

            if data_train is None:
                data_train = x_train
                label_train = y_train
            else:
                data_train = torch.cat((data_train, x_train), 0)
                label_train = torch.cat((label_train, y_train), 0)



    return data_train, label_train


# normalization
def normalization(img):
    img_ = np.array(img).copy()
    b, c, d, h, w = img.shape
    for i in range(b):
        for j in range(c):
            for k in range(d):
                tempImg = img_[i, j, k, :, :]
                pMax = np.percentile(tempImg, 99)
                pMin = np.percentile(tempImg, 1)
                tempImg[tempImg > pMax] = pMax
                tempImg[tempImg < pMin] = pMin
                tempImg = (tempImg - tempImg.min()) / (tempImg.max() - tempImg.min())
                img_[i, j, k, :, :] = tempImg

    return torch.tensor(img_)


# define train data set
class TraindataSet(Dataset):
    def __init__(self, train_features, train_labels, transform=None):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)
        self.transform = transform

    def __getitem__(self, index):

        if self.transform is not None:
            x, xRot_mat, xTrans_mat, xEtFlag, xEtIndices, xVFlipFlag, xHFlipFlag = self.transform(self.x_data[index])
            y, yRot_mat, yTrans_mat, yEtFlag, yEtIndices, yVFlipFlag, yHFlipFlag = self.transform(self.y_data[index],
                                                                                                  rot_mat=xRot_mat,
                                                                                                  trans_mat=xTrans_mat,
                                                                                                  etFlag=xEtFlag,
                                                                                                  indices=xEtIndices,
                                                                                                  vFlipFlag=xVFlipFlag,
                                                                                                  hFlipFlag=xHFlipFlag)
            return torch.Tensor(x), torch.Tensor(y)
        else:
            return torch.Tensor(self.x_data[index]), torch.Tensor(self.y_data[index])

    def __len__(self):
        return self.len


def randomTransform(img, etTrhd=0.5, vFlipTrhd=0.5, hFlipTrhd=0.5, rot_mat=None, trans_mat=None, etFlag=None,
                    indices=None, vFlipFlag=None, hFlipFlag=None):
    img_ = np.array(img).copy()
    c, d, h, w = img.shape

    # translate and rotate parameters
    if rot_mat is None:
        translateH = random.uniform(-0.1, 0.1)
        translateW = random.uniform(-0.1, 0.1)
        rotatAngle = random.uniform(-90, 90)
        scale = random.uniform(0.8, 1.2)
        # translate and rotate
        center = (w // 2, h // 2)
        rot_mat = cv.getRotationMatrix2D(center, rotatAngle, scale)
        trans_mat = np.float64([[1, 0, w * translateW], [0, 1, h * translateH]])

    for j in range(c):
        for k in range(d):
            img_[j, k, :, :] = cv.warpAffine(img_[j, k, :, :], rot_mat, (w, h))
            img_[j, k, :, :] = cv.warpAffine(img_[j, k, :, :], trans_mat, (w, h))

    # elastic Transform
    if etFlag is None:
        etFlag = random.uniform(0, 1)
    if etFlag < etTrhd:
        if indices is None:
            # elastic transform
            seed = np.random.randint(1, 100)
            randomState = np.random.RandomState(seed)
            elasticAlpha = random.uniform(10, 20)
            elasticSigma = random.uniform(20, 30)
            dx = gaussian_filter((randomState.rand(h, w) * 2 - 1), elasticSigma)
            dy = gaussian_filter((randomState.rand(h, w) * 2 - 1), elasticSigma)
            # elastic transform
            maxNorm = np.max((dx ** 2 + dy ** 2) ** 0.5)
            dx = elasticAlpha * dx / maxNorm
            dy = elasticAlpha * dy / maxNorm
            y, x = np.meshgrid(np.arange(w), np.arange(h))
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        for j in range(c):
            for k in range(d):
                img_[j, k, :, :] = map_coordinates(img_[j, k, :, :], indices, order=1, mode='nearest').reshape([h, w])

    # Vertical flip Transform
    if vFlipFlag is None:
        vFlipFlag = random.uniform(0, 1)
    if vFlipFlag < vFlipTrhd:
        for j in range(c):
            for k in range(d):
                img_[j, k, :, :] = cv.flip(img_[j, k, :, :], 0)

    # Horizontal flip Transform
    if hFlipFlag is None:
        hFlipFlag = random.uniform(0, 1)
    if hFlipFlag < hFlipTrhd:
        for j in range(c):
            for k in range(d):
                img_[j, k, :, :] = cv.flip(img_[j, k, :, :], 1)

    return torch.tensor(img_), rot_mat, trans_mat, etFlag, indices, vFlipFlag, hFlipFlag