# 1. Cut off last work of a sentence
sentence = "Hello I am here"
# Use rsplit() - reverse split
# , 1 - make a single split
# ' ' - split on spaces
# [0] - take first
sentence = sentence.rsplit(' ', 1)[0]


# 2. List contents of a folder
import os
os.listdir("path") # returns list