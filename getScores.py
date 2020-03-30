import sys
import codecs
import re
import regex
from subprocess import call

fIn = codecs.open(sys.argv[1], "r", "utf-8")

maxScore = 0.0
save = False
maxCheckpoint = ""
for i in fIn:

    if("Validation accuracy" in i):
        parts = i.split(" ")
        score = float(parts[-1])
        if(score > maxScore or maxCheckpoint == parts[-1].strip()): #Catch duplicate runs
            maxScore = score
            save = True
    elif("Saving checkpoint" in i and save):
        save = False
        parts = i.split(" ")
        maxCheckpoint = parts[-1].strip()
print("Max score: ", maxScore, " ", maxCheckpoint.strip())

parts = maxCheckpoint.split("/")
exptName = "Results/dev/" + parts[-3] + "/" + parts[-2] + "/" + parts[-1].replace(".pt",".results")
print(exptName)
devFile = "morphData/dev/" + parts[-3] + "-dev" + ".src"

call(["python3", "translate.py", "-model", maxCheckpoint, "-src", devFile, "-output", exptName, "-replace_unk"])



fIn.close();
