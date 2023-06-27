f0 = open('test.txt')
lines = f0.readlines()
f0.close()
newlines = []

for i in range(len(lines)):
    newlines.append(lines[i].strip().split(' ')[0]+'\n')

f = open('test1.txt', 'w')

f.writelines(newlines)
f.close()