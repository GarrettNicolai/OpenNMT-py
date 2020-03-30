import sys
import codecs
import re
import regex
from subprocess import call
import random

fLow = codecs.open("lowGPU.expts", "w", "utf-8")
fMed = codecs.open("medGPU.expts", "w", "utf-8")
fHigh = codecs.open("highGPU.expts", "w", "utf-8")


seeds = [random.randint(0,1000000) for k in range(0,20)]

for j in ["arabic", "basque", "english", "finnish", "german", "navajo", "persian", "sanskrit", "turkish", "zulu", "hindi", "urdu", "georgian"]:
    for k in ["teacher", "student", "random", "dist"]:
        for m in seeds:
            fLow.write("\t".join([j, "low", str(m), k]) + "\n") 
            fMed.write("\t".join([j, "medium", str(m), k]) + "\n") 
            fHigh.write("\t".join([j, "high", str(m), k]) + "\n") 



fLow.close();
fMed.close();
fHigh.close();
