import numpy as np

f = open('pairs.txt', 'w')

lines = []

testf = open('test.txt')
test_imgs = testf.readlines()
testf.close()

person = [test_imgs[40 * i].split()[1] for i in range(80)]

root = '/root/autodl-tmp/generation/ROI/'
lines.append('10\t300\n')

for _ in range(10):
    # 400 pairs from same hands
    h = np.random.randint(0, 80, 400)  # num of hand

    i1 = np.random.randint(0, 40, 400)   # get 2 images from the hand
    i2 = (i1 + np.random.randint(1, 40, 400)) % 40

    for k in range(400):
        line = person[h[k]] + '\t'
        line += test_imgs[40 * h[k] + i1[k]].strip().split()[0] + '\t'
        line += test_imgs[40 * h[k] + i2[k]].strip().split()[0] + '\n'
        lines.append(line)

    # 400 pairs from different hands
    # get 2 hands
    h1 = np.random.randint(0, 80, 400)
    # make sure two hands are from different people and the same side
    h2 = (h1 + np.random.randint(1, 40, 400) * 2) % 80


    i1 = np.random.randint(0, 40, 400)   # get 2 images from the hand
    i2 = np.random.randint(0, 40, 400)

    for k in range(400):
        line = person[h1[k]] + '\t'
        line += test_imgs[40 * h1[k] + i1[k]].strip().split()[0] + '\t'
        line += person[h2[k]] + '\t'
        line += test_imgs[40 * h2[k] + i2[k]].strip().split()[0] + '\n'
        lines.append(line)

f.writelines(lines)
f.close()