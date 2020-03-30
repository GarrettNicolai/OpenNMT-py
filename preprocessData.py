import sys
import codecs
import re
import regex

print(sys.argv)
fIn = codecs.open(sys.argv[1], "r", "utf-8")
fOutSrc = codecs.open(sys.argv[2], "w", "utf-8")
fOutTrg = codecs.open(sys.argv[3], "w", "utf-8")

for i in fIn:
    parts = i.strip().split('\t')
    lemma = parts[0].replace(" ", "_")
    tags = parts[2].split(";")
    inflected = parts[1].replace(" ", "_")
    fOutSrc.write(" ".join(regex.findall(r'\X', lemma, regex.U)).strip() + " TAG=" + " TAG=".join(tags) + "\n")
    fOutTrg.write(" ".join(regex.findall(r'\X', inflected, regex.U)).strip() + "\n")

fIn.close();
fOutSrc.close();
fOutTrg.close();
