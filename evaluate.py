import sys
import codecs
import re
import regex
from subprocess import call
from nltk.metrics.distance import edit_distance
outname = sys.argv[1].replace("results","errors")
#print(outname)
fIn = codecs.open(sys.argv[1], "r", "utf-8")
fGold = codecs.open(sys.argv[2], "r", "utf-8")
fOut = codecs.open(outname, "w", "utf-8")


total = 0.0
correct = 0.0

totalL5 = 0.0
totalL10 = 0.0
totalG10 = 0.0
totalID = 0.0

correctL5 = 0.0
correctL10 = 0.0
correctG10 = 0.0
correctID = 0.0

ED = 0.0
EDL5 = 0.0
EDL10 = 0.0
EDG10 = 0.0

for pred in fIn:
    pred = pred.replace(" ","").strip()
    gold = fGold.readline().strip()
    gold = gold.replace(" ","_")
    goldParts = gold.split("\t")

    lemma = goldParts[0]
    inflection = goldParts[1]
    tag = goldParts[2]
    ed = edit_distance(inflection, pred)

    lenLemma = len(lemma)
    total += 1
    if(lemma == inflection):
        totalID += 1
        if(pred == inflection):
            correctID += 1
    if lenLemma <= 5:
        totalL5 += 1
        ED += float(ed) / len(inflection)
        EDL5 += float(ed) / len(inflection)

        if(pred == inflection):
            correctL5 += 1
            correct += 1
            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "S+\n")
        else:
            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "S-\n")
            
    elif lenLemma <= 10:
        totalL10 += 1
        ED += float(ed) / len(inflection)
        EDL10 += float(ed) / len(inflection)

        if(pred == inflection):
            correctL10 += 1
            correct += 1
            ED += float(ed) / len(inflection)
            EDL10 += float(ed) / len(inflection)

            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "M+\n")
        else:
            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "M-\n")
            
    else:
        totalG10 += 1
        ED += float(ed) / len(inflection)
        EDG10 += float(ed) / len(inflection)

        if(pred == inflection):
            correctG10 += 1
            correct += 1
            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "L+\n")
        else:
            fOut.write(lemma + "\t" + tag + "\t" + inflection + "\t" + pred + "\t" + "L-\n")
            
if total == 0:
    accuracy = 0
    ED = 0.0
else:
    ED /= total
    accuracy = correct / total
            
if totalL5 == 0:
    accuracyL5 = 0.0
    EDL5 = 0.0
else:
    EDL5 /= totalL5
    accuracyL5 = correctL5 / totalL5
            
if totalL10 == 0:
    accuracyL10 = 0
    EDL10 = 0.0
else:
    EDL10 /= totalL10
    accuracyL10 = correctL10 / totalL10
            
if totalG10 == 0:
    accuracyG10 = 0
    EDG10 = 0
else:
    EDG10 /= totalG10
    accuracyG10 = correctG10 / totalG10

if totalID == 0:
    accuracyID = 0
else:
    accuracyID = correctID / totalID

print("Total: ", total)
print("Correct: ", correct)
print("Accuracy: ", accuracy)
print("ED: ", ED)


print("Total <= 5: ", totalL5)
print("Correct <= 5: ", correctL5)
print("Accuracy <= 5: ", accuracyL5)
print("ED <= 5: ", EDL5)

print("Total <= 10: ", totalL10)
print("Correct <= 10: ", correctL10)
print("Accuracy <= 10: ", accuracyL10)
print("ED <= 10: ", EDL10)

print("Total > 10: ", totalG10)
print("Correct > 10: ", correctG10)
print("Accuracy > 10: ", accuracyG10)
print("ED > 10: ", EDG10)

print("Total ID: ", totalID)
print("Correct ID: ", correctID)
print("Accuracy ID: ", accuracyID)

fIn.close();
