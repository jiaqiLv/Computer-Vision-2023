
from PIL import Image
import torch
import numpy as np
from model import MobileFaceNet
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

file_path = './dataset/test1.txt'
weight_path = '/root/MobileFaceNet2/saved_models/MobileFaceNet_v2_20220614_231536/best.pth'

with open(file_path) as f:
    lines = f.readlines()

imgs = [line.strip() for line in lines]

net = MobileFaceNet()
net.load_state_dict(torch.load(weight_path))
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.functional.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_imgs = [transform(Image.open(img)) for img in imgs]

test_loader = torch.utils.data.DataLoader(test_imgs, batch_size = 1024, pin_memory = True, shuffle=False, num_workers=24)

feats = None
net = nn.DataParallel(net, device_ids = [0, 1])
net.cuda()
net.eval()
with torch.no_grad():
    for data in test_loader:
        feat = net(data.to('cuda')).cpu()
        feat = nn.functional.normalize(feat).numpy()
        if feats is not None:
            feats = np.concatenate((feats, feat), 0)
        else:
            feats = feat
        
accuracy = 0
reg = np.array([feats[i * 40] for i in range(80)]).T

for i in range(80):
    cnt = 0
    for j in range(1, 40):
        if i==0 and j==1:
            print(reg.shape)
        scores = np.sum(reg * np.expand_dims(feats[40 * i + j], 1), axis=0)
        if np.sum(scores > scores[i]) == 0:
            cnt += 1
        else:
            print(scores)
            print(i)

    accuracy += cnt / 39

accuracy /= 80

print('Top-1 accuracy: %.3f' % accuracy)
