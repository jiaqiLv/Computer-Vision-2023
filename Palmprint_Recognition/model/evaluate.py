import sys
# import caffe
import os
import numpy as np
import scipy.io
import model
import os
import torch.utils.data
from dataloader import PalmTestSet
import torch.nn.functional as F
import argparse

def parseList(root):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = p[1]
            nameR = p[2]
            fold = i // 800
            flag = 1
        elif len(p) == 4:
            nameL = p[1]
            nameR = p[3]
            fold = i // 800
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    # print(nameLs)
    return [nameLs, nameRs, folds, flags]



def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def getThresholdByFAR(scores, flags, step=1e-4, FAR=1e-4):
    threshold = -1
    while threshold < 1:
        fp = np.sum(scores[flags == -1] > threshold)
        if fp / np.sum(flags == -1) < FAR:
            break
        threshold += step

    tp = np.sum(scores[flags == 1] > threshold)

    return threshold, tp / np.sum(flags == 1)


def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        print('   threshold = %.3f' % threshold)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs



def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.MobileFaceNet()
    if gpu:
        net = net.cuda()
    if resume:
        weights = torch.load(resume)
        net.load_state_dict(weights)
    net.eval()
    nl, nr, folds, flags = parseList(lfw_dir)
    lfw_dataset = PalmTestSet(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,
                                              shuffle=False, num_workers=24, drop_last=False)

    featureLs = None
    featureRs = None

    for data in lfw_loader:
        imgl, imgr = data
        with torch.no_grad():
            featureL = net(imgl.cuda()).data.cpu()
            featureR = net(imgr.cuda()).data.cpu()
        # normalize the features
        featureL = F.normalize(featureL).numpy()
        featureR = F.normalize(featureR).numpy()

        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)

    result = {'fl': featureLs, 'fr': featureRs,'fold': folds, 'flag': flags}
    # save tmp_result
    scipy.io.savemat(feature_save_dir, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--resume', type=str, default='/root/MobileFaceNet2/saved_models/MobileFaceNet_v2_20220614_231536/best.pth',
                        help='The path pf save model')
    parser.add_argument('--feature_save_dir', type=str, default='./result/eval_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()


    # getFeatureFromCaffe()
    getFeatureFromTorch('dataset', args.feature_save_dir, args.resume)

    result = scipy.io.loadmat(args.feature_save_dir)

    fold = result['fold']
    flags = result['flag']
    featureLs = result['fl']
    featureRs = result['fr']
    scores = np.sum(np.multiply(featureLs, featureRs), 1)
    flags = np.squeeze(flags)

    for far in [1e-1, 1e-2, 1e-3, 1e-4]:
        thresh, tpr = getThresholdByFAR(scores, flags, 1e-4, far)
        print('far = %.4f, thresh = %.4f, tpr = %.4f' % (far, thresh, tpr))

    
    # ACCs = evaluation_10_fold(args.feature_save_dir)
    # for i in range(len(ACCs)):
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
