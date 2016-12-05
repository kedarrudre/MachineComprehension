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

# data
trainData = ('/Users/msingh/cs221/project/mctDataSet/MCTest/mc160.train.tsv', \
                 '/Users/msingh/cs221/project/mctDataSet/MCTest/mc160.train.ans')

devData = ('/Users/msingh/cs221/project/mctDataSet/MCTest/mc160.dev.tsv', \
               '/Users/msingh/cs221/project/mctDataSet/MCTest/mc160.dev.ans')


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

for story in dev.stories+train.stories:
    if verbose:
        print story.name
    countStories += 1
    ans = baseLineAlgo.predict(story)
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
            

# compare and print out statistics
if verbose:
    print '**** Statistics ****'

def read_answers(f):
    fin = open(f, 'r')
    answers = fin.readlines()
    fin.close()
    answers = map(lambda x: x.rstrip().split('\t'), answers)
    return reduce(lambda x, y: x + y, answers)

correctAnswers = np.array(read_answers(devData[1]) + read_answers(trainData[1]))
myAnswers = np.array(reduce(lambda x, y: x + y, answers))
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
                                         
    
