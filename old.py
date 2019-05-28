import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
import json

words = Counter()

# Count the words in the text file
with open('./text.txt', 'r') as text:
    content = text.read().split(' ')
    for word in content:
        word = re.sub(r'[^\w+]', '', word).lower()
        if word in words.keys():
            words[word] += 1
        else:
            words[word] = 1

# Dump them for use later
with open('./results.json', 'w') as data:
    data.write(json.dumps(words.most_common(), indent=4))

# Extract sorted words and frequencies from counter object
sorted_words = [word[0] for word in words.most_common()]
sorted_freq = [word[1] for word in words.most_common()]
ranks = list(range(1, len(sorted_words)+1))

# # Find nearest entry of numpy array to some value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# # Generate a logarithmic range from 1 to the end of the array
log_range = np.geomspace(1, len(sorted_words) + 1, num=3000)

# First find all the nearest ranks to the values from that logarithmic range
# Then vectorize those values so that you get mostly large values
# Then get all the unique ones
# Then get all the frequencies from those ranks
vectorizer = np.vectorize(lambda e: len(sorted_freq) + 1 - e)
new_points = np.unique(vectorizer(np.array([find_nearest(ranks, rank) for rank in log_range])))
new_freqs = [sorted_freq[rank-1] for rank in new_points]

# # Fit a curve to 
slope, intercept = np.polyfit(np.log(new_points), np.log(new_freqs), 1)

print(f'y ={slope}x + {intercept}')

# Create the plot
fig = plt.figure()

plt.xscale('log')
plt.yscale('log')

x = np.linspace(1, 1000, 2)
y = sorted_freq[0] * np.power(x, slope)

plt.plot(x, y)
plt.scatter(ranks, sorted_freq)

plt.show()
plt.savefig('plot.png')

def zipf_freq(rank, slope, most_freq_word):
    return math.floor(most_freq_word * math.pow(rank, slope))

rank = int(input("Input a rank: "))
print(f"You chose the word {sorted_words[rank-1]} with frequency {sorted_freq[rank-1]}")
print(f"Calculated frequency is: {zipf_freq(rank, slope, sorted_freq[0])}")
