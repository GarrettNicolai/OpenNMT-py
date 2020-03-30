import sys
import codecs
import re
import regex
from subprocess import call

fIn = codecs.open(sys.argv[1], "r", "utf-8")
fIn2 = codecs.open(sys.argv[2], "r", "utf-8")

train = {}
L5 = 0
L10 = 0
G10 = 0

for i in fIn:
    parts = i.strip().split("\t")
    train[parts[0]] = True


count = 0
averageL = 0.0
averageI = 0.0
ratio = 0.0
for i in fIn2:
    parts = i.strip().split("\t")
    if(parts[0] in train):
        count += 1
    if len(parts[0]) <= 5:
        L5 += 1
    elif len(parts[0]) <= 10:
        L10 += 1
    else:
        G10 += 1
    averageL += len(parts[0])
    averageI += len(parts[1])
    ratio += (len(parts[0]) / float(len(parts[1])))
print("COUNT: ", count)
print("L5: ", L5)
print("L10: ", L10)
print("G10: ", G10)
print("LAve: ", averageL)
print("IAve: ", averageI)
print("Ratio: ", ratio)

fIn.close();
fIn2.close();
