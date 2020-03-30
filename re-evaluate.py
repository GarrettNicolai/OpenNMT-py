import sys
import codecs
import re
import regex
import subprocess

fIn = codecs.open(sys.argv[1], "r", "utf-8")

maxScore = 0.0
save = False
maxCheckpoint = ""
files = {}
for i in fIn:

    parts = i.strip().split("/")
    lang = parts[2]
    fileName = lang + ".reEval"
    if(lang not in files):
        files[lang] = codecs.open(fileName, "w", "utf-8")
    outFile = files[lang]
    expt = parts[-1]
    exptParts = expt.split("_")
    exptType = exptParts[0]
    setting = parts[3]
    result = subprocess.run(["echo", lang + "-" + exptType + "-" + setting + "-" + exptParts[2]], stdout=subprocess.PIPE, universal_newlines=True)
    outFile.write(result.stdout)
    result = subprocess.run(["python3", "evaluate.py", i.strip(), "morphData/dev/" + lang + "-dev"], stdout=subprocess.PIPE, universal_newlines=True) 
    #print(result)
    outFile.write(result.stdout)



fIn.close();
for i in files:
    fOut = files[i]
    fOut.close()
