import sys
import codecs
import re
import regex
import subprocess
import numpy
fIn = codecs.open(sys.argv[1], "r", "utf-8")
fOut = codecs.open(sys.argv[2], "w", "utf-8")


scores = {}
filename = ""
fileParts = []
current = {}
for i in fIn:
    if(i.strip() == "^$"):
        continue
    if not i[0].isupper():
        filename = i.strip()
        fileParts = filename.split("-")
        if(fileParts[0] not in scores):
            scores[fileParts[0]] = {}
        if(fileParts[2] not in scores[fileParts[0]]):
            scores[fileParts[0]][fileParts[2]] = {}
        if(fileParts[1] not in scores[fileParts[0]][fileParts[2]]):
            scores[fileParts[0]][fileParts[2]][fileParts[1]] = {}
            current = scores[fileParts[0]][fileParts[2]][fileParts[1]]
            current["ED"] = {}
            current["Accuracy"] = {}
            
            current["ED"]["L5"] = []
            current["ED"]["L10"] = []
            current["ED"]["G10"] = []
            current["ED"]["ID"] = []
            current["ED"]["All"] = []

            current["Accuracy"]["L5"] = []
            current["Accuracy"]["L10"] = []
            current["Accuracy"]["G10"] = []
            current["Accuracy"]["ID"] = []
            current["Accuracy"]["All"] = []


    else:
        scoreParts = i.strip().split(" ")
        scoreType = ""
        filterType = ""

        if("Accuracy" in i):
            scoreType = "Accuracy"
        elif("ED" in i):
            scoreType = "ED"
        else:
            continue
        if("<= 5" in i):
            filterType = "L5"
        elif("<= 10" in i):
            filterType = "L10"
        elif("> 10" in i):
            filterType = "G10"
        elif "ID" in i:
            filterType = "ID"
        else:
            filterType = "All"
        current[scoreType][filterType].append(float(scoreParts[-1]))

Averages = {}
Variances = {}
for i in sorted(scores.keys()): #languages
    fOut.write(i.upper() + "\t")
    for j in scores[i]:
        if(j not in Averages): #low, med, high
            Averages[j] = {}
            Variances[j] = {}

        for k in scores[i][j]: #exptType
            if(k not in Averages[j]):
                Averages[j][k] = {}
                Variances[j][k] = {}
            for m in scores[i][j][k]: #Accuracy or ED
                if(m not in Averages[j][k]):
                    Averages[j][k][m] = {}
                    Variances[j][k][m] = {}

                for n in scores[i][j][k][m]: #L5, L10, G10, All
                    if(n not in Averages[j][k][m]):
                        Averages[j][k][m][n] = []
                        Variances[j][k][m][n] = []
                    if(len(scores[i][j][k][m][n]) != 0):
                        currentAverage = numpy.mean(scores[i][j][k][m][n])
                        currentVariance = numpy.var(scores[i][j][k][m][n])
                    else:
                        currentAverage = 0.0
                        currentVariance = 0.0
                    Averages[j][k][m][n].append(currentAverage)
                    Variances[j][k][m][n].append(currentVariance)
                    #fOut.write(i + " " + j + " " + k + " " + m + " " + n + " Average: " + str(currentAverage) + "\n")
                    #fOut.write(i + " " + j + " " + k + " " + m + " " + n + " Variance: " + str(currentVariance) + "\n")
fOut.write("OVERALL AVERAGE\n")
#print(Averages["finnish"]["high"]["teacher"])
#for i in Averages:
for j in Averages:
    for k in Averages[j]:
        for m in Averages[j][k]:
            for n in Averages[j][k][m]:
                fOut.write(j + " " + k + " " + m + " " + n + " AVERAGE: ")

                for i in sorted(scores.keys()):
                    if(len(scores[i][j][k][m][n]) == 0):
                        fOut.write("0.0" + "\t")
                    else:
                        fOut.write(str(numpy.mean(scores[i][j][k][m][n])) + "\t")
                    #fOut.write(str(numpy.var(scores[i][j][k][m][n])) + "\t")

                    #currentVariance = numpy.mean(Variances[i][j][k][m][n])
                    #fOut.write("VARIANCE: " + j + " " + k + " " + m + " " + n + " : " + str(currentVariance) + "\n")
                currentAverage = numpy.mean(Averages[j][k][m][n])
                fOut.write(str(currentAverage) + "\n")
                
                fOut.write(j + " " + k + " " + m + " " + n + " VARIANCE: ")
                for i in sorted(scores.keys()):
                    #fOut.write(str(numpy.mean(scores[i][j][k][m][n])) + "\t")
                    if(len(scores[i][j][k][m][n]) == 0):
                        fOut.write("0.0" + "\t")
                    else:
                        fOut.write(str(numpy.var(scores[i][j][k][m][n])) + "\t")

                    #currentAverage = numpy.mean(Averages[i][j][k][m][n])
                    #fOut.write("AVERAGE: " + j + " " + k + " " + m + " " + n + " : " + str(currentAverage) + "\n")
                currentVariance = numpy.mean(Variances[j][k][m][n])
                fOut.write(str(currentVariance) + "\n")


fIn.close()
fOut.close()
