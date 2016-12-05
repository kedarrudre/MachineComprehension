#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import collections
import re
import mcaAlgorithms
from functools import reduce
import numpy as np


## Verbosity/logfile
verbose = 1
logFile = 'mca.log'
genInput = 1

# data
trainData = ('data/mc160.train.tsv', 'data/mc160.train.ans')
devData = ('data/mc160.dev.tsv', 'data/mc160.dev.ans')


if verbose:
    print '**** Reading data ****'
train = mcaAlgorithms.MCTReadData(trainData[0], verbose)
dev = mcaAlgorithms.MCTReadData(devData[0], verbose)

## Base Line algorithm
baseLineAlgo = mcaAlgorithms.MCTSlidingWindow(verbose)
if verbose > 0:
    print 'Size of stories, dev = %s, train = %s' %(len(train.stories), len(dev.stories))

if verbose:
    print '**** training ****'
baseLineAlgo.fit(dev.stories+train.stories)

# predict answers
answers = []
qtypes = []
countStories = 0
if verbose:
    print '**** predicting answers ****'

def read_answers(f):
    fin = open(f, 'r')
    answers = fin.readlines()
    fin.close()
    answers = map(lambda x: x.rstrip().split('\t'), answers)
    return reduce(lambda x, y: x + y, answers)
    
genfile = open("inputgen.txt", "w")
storyIdx = 0
stories = dev.stories+train.stories
correctAnswers = np.array(read_answers(devData[1]) + read_answers(trainData[1]))
for story in stories:
    if verbose:
        print story.name
    countStories += 1
    ans, allScores = baseLineAlgo.predict(story, genfile)
    answers.append(ans)
    for id, question in enumerate(story.rawQuestions):
        if question[0] == 'multiple':
            qtypes.append(1)
        else:
            qtypes.append(0)
        if verbose >= 3:
            print question
            print story.rawAnswers[id]
            print ans[id]
            print '\n'
    
    tempMap = { 'A':0, 'B':1, 'C':2, 'D':3}
    # genfile.write("STORY: " + story.name + "\n")
    for scoreIdx, score in enumerate(allScores):        
        # if scoreIdx % 4 == 0:
            # genfile.write("\n")
        corrAnsId = tempMap[correctAnswers[storyIdx*4 + (scoreIdx/4)]]
        yVal = '0'
        if scoreIdx % 4 == corrAnsId:
            yVal = '1'
        ansMatched = '0'
        if correctAnswers[storyIdx*4 + (scoreIdx/4)] == ans[scoreIdx/4]:
            ansMatched = '1'
        # genfile.write(str(score) + "\t" + correctAnswers[storyIdx*4 + (scoreIdx/4)] + "\t" + \
        # str(corrAnsId) +  "\t" + yVal + "\n")
        genfile.write(str(score) + "\t" + yVal + "\n")
        
        #genfile.write(str(score) + "\t" + yVal + "\n")
    storyIdx += 1
genfile.close()

# compare and print out statistics
if verbose:
    print '**** Statistics ****'

correctAnswers = np.array(read_answers(devData[1]) + read_answers(trainData[1]))
myAnswers = np.array(reduce(lambda x, y: x + y, answers))
print "CORRECT ANSWERS: ", correctAnswers
print "MY ANSWERS: ", myAnswers
assert len(myAnswers) == len(correctAnswers)
correct = np.sum(myAnswers == correctAnswers)
total = len(myAnswers)

multiples = np.array(qtypes) == [1]*len(correctAnswers)
num_m = float(np.sum(multiples))
num_s = float(np.sum(~multiples))

assert num_m + num_s == total

print 'All accuracy [%d]: %.4f' %(total, float(correct)/float(total))
print 'Single accuracy [%d]: %.4f' %(num_s, np.sum(myAnswers[~multiples] == correctAnswers[~multiples])/num_s)
print 'Multiple accuracy [%d]: %.4f' %(num_m, np.sum(myAnswers[multiples] == correctAnswers[multiples])/num_m)
                                         
    
