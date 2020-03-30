import re
import sys
from math import log
import codecs

fIn = codecs.open(sys.argv[1], "r", "utf-8")

#Function to compute the base-2 log of a floating point number
def log2(number):
    return log(number) / log(2)

cleaner = re.compile('[^a-z]+')

letter_frequency = {}


text_len = 0
text_len2 = 0.0
for i in fIn:
    parts = i.strip().split("\t")
    parts[1] = parts[1].replace(" ","_")
    text_len += len(parts[1])
    text_len2 += len(parts[0])
    for letter in parts[1]:
        if letter in letter_frequency:
            letter_frequency[letter] += 1
        else:
            letter_frequency[letter] = 1

length_sum = 0.0
for letter in letter_frequency:
    prob = float(letter_frequency[letter]) / text_len
    length_sum += prob * log2(prob)

print('Entropy: %f bits per character\n' % (-length_sum))
print('Average Lemma: %f characters\n' % (text_len2 / 10000))


