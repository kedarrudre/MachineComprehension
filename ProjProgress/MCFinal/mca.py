#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import collections
import re
import mcaAlgorithms
from functools import reduce
import numpy as np
import argparse
import util

parser = argparse.ArgumentParser(description='Machine Comprehension Algorithm by Manish Singh/Kedar Rudre')
parser.add_argument("-algo", help='Possible algorithms: [slidingWindow, linearRegression]', default='slidingWindow')
parser.add_argument("-stories", help="stories in tsv format; can be added multiple times", default=None, required=True, action='append')
parser.add_argument("-answers", help="answers in tsv format; can be added multiple times", default=None, required=True, action='append')
parser.add_argument("-verbose", type=int, help="increase output verbosity", default=0)
parser.add_argument("-eta", type=float, help="SGD step", default=0.001)
parser.add_argument("-T",  type=int, help="SGD iterations of total samples", default=3)
parser.add_argument("-Lambda", type=float, help="Lamda", default=0.1)
parser.add_argument("-delta", type=float, help="delta for gradient calculation", default=0.0001)
parser.add_argument("-cccp_itr_count", type=int, help="Number of CCCP iterations", default=20)
parser.add_argument("-corefDataDir", type=str, help="stories after coreference resolved", default='data/mctDataSetAfterCoref')
parser.add_argument("--notrain", help='dont train, read weights from file <weights.txt>', action='store_true')
parser.add_argument("--startFromExistingWeights", help='read weights from file <weights.txt> to start', action='store_true')
parser.add_argument("--useNltkStopWords", help='use nltk english stop words', action='store_true')
parser.add_argument("--useCorefFeatures", help='use coreference features', action='store_true')

parser.add_argument("-sum_tfidf_on", type=int, help="0: don't use sum_tfidf, 1: use sum_tfidf", default=1)
parser.add_argument("-baseline_score_on", type=int, help="0: don't use baseline_score, 1: use baseline_score", default=0)
parser.add_argument("-sent_score_on", type=int, help="0: don't use sent_score, 1: use sent_score", default=1)
parser.add_argument("-sent_perStory_score_on", type=int, help="0: don't use sent_perStory_score, 1: use sent_perStory_score", default=1)
parser.add_argument("-sent2_perStory_score_on", type=int, help="0: don't use sent2_perStory_score, 1: use sent2_perStory_score", default=1)
parser.add_argument("-sent3_perStory_score_on", type=int, help="0: don't use sent3_perStory_score, 1: use sent3_perStory_score", default=0)
parser.add_argument("-question_negation_on", type=int, help="0: don't use question_negation, 1: use question_negation", default=0)
parser.add_argument("-length", type=int, help="0: don't use length, 1: use length", default=1)
parser.add_argument("-question_type_on", type=int, help="0: don't use question_type, 1: use question_type", default=0)

args = parser.parse_args()

def slidingWindow():
    if args.verbose:
        print '**** Reading data ****'

    stories = []
    correctAnswers = []
    def read_answers(f):
        fin = open(f, 'r')
        answers = fin.readlines()
        fin.close()
        answers = map(lambda x: x.rstrip().split('\t'), answers)
        return answers

    for _ in args.stories:
        stories.append(mcaAlgorithms.MCTReadData(_, args))
    
    for _ in args.answers:
        correctAnswers += read_answers(_)
        
    ## Base Line algorithm
    baseLineAlgo = mcaAlgorithms.MCTSlidingWindow(args)

    if args.verbose > 0:
        print '**** training ****'
        for i, _ in enumerate(args.stories):
            print 'Size of stories in %s is %s' %(_, len(stories[i].stories))

    all_stories = []
    for _ in stories:
        all_stories += _.stories
    baseLineAlgo.fit(all_stories)
    
    # predict answers
    answers = []
    qtypes = []
    countStories = 0

    if args.verbose:
        print '**** predicting answers ****'

    for _ in range(len(args.stories)):
        for story in stories[_].stories:
            if args.verbose:
                print story.name
            countStories += 1
            ans = baseLineAlgo.predict(story)
            answers.append(ans)
            for id, question in enumerate(story.rawQuestions):
                if question[0] == 'multiple':
                    qtypes.append(1)
                else:
                    qtypes.append(0)
                if args.verbose >= 3:
                    print question
                    print story.rawAnswers[id]
                    print ans[id]
                    print '\n'

    # compare and print out statistics
    if args.verbose:
        print '**** Statistics ****'

    myAnswers = np.array(reduce(lambda x, y: x + y, answers))
    correctAnswers = np.array(reduce(lambda x, y: x + y, correctAnswers))
    
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

def linearRegression():
    if args.verbose:
        print '**** Reading data ****'

    stories = []
    for i in range(len(args.stories)):
        myStoryObject = mcaAlgorithms.MCTReadData(args.stories[i], args)
        fin = open(args.answers[i], 'r')
        answers = map(lambda x: x.rstrip().split('\t'), fin.readlines())
        fin.close()
        for i, myStory in enumerate(myStoryObject.stories):
#            print 'setting answers for story %s' %(answers[i])
            myStory.setCorrectAnswer(map(util.answerLetters2Num,answers[i]))
        stories.append(myStoryObject)

    fin.close()
    
    ## Linear Regression algorithm
    linearRegressionAlgo = mcaAlgorithms.MCTLinearRegression(args)

    # update parameters
    linearRegressionAlgo.updateParameter('eta', args.eta)
    linearRegressionAlgo.updateParameter('T', args.T)
    linearRegressionAlgo.updateParameter('lambda', args.Lambda)
    linearRegressionAlgo.updateParameter('delta', args.delta)
    linearRegressionAlgo.updateParameter('number_of_cccp_iterations', args.cccp_itr_count)

    if args.notrain:
        linearRegressionAlgo.updateParameter('read_weights_from_file', True)
    if args.startFromExistingWeights:
        linearRegressionAlgo.updateParameter('startFromExistingWeights', True)
    
    if args.verbose > 0:
        print '**** training ****'
        for i, _ in enumerate(args.stories):
            print 'Size of stories %s is %s' %(_, len(stories[i].stories))

    # train
    all_stories = []
    for _ in stories:
        all_stories += _.stories
    
    linearRegressionAlgo.calc_icounts(all_stories)
    linearRegressionAlgo.fit(all_stories)

    # predict
    answers = []
    if args.verbose:
        print '**** predicting answers ****'

    answers.append(linearRegressionAlgo.predict(_.stories))

    correctAnswers = []
    qtypes = []
    countStories = 0

    def read_answers(f):
        fin = open(f, 'r')
        answers = fin.readlines()
        fin.close()
        answers = map(lambda x: x.rstrip().split('\t'), answers)
        return answers

    for _ in args.answers:
        correctAnswers += read_answers(_)

    for _ in range(len(args.stories)):
        for story in stories[_].stories:
            if args.verbose:
                print story.name
            countStories += 1
            for id, question in enumerate(story.rawQuestions):
                if question[0] == 'multiple':
                    qtypes.append(1)
                else:
                    qtypes.append(0)

    myAnswers = np.array(reduce(lambda x, y: x + y, answers))
    correctAnswers = np.array(reduce(lambda x, y: x + y, correctAnswers))
    
    assert len(myAnswers) == len(correctAnswers)
    correct = np.sum(myAnswers == correctAnswers)
    total = len(myAnswers)

    multiples = np.array(qtypes) == [1]*len(correctAnswers)
    num_m = float(np.sum(multiples))
    num_s = float(np.sum(~multiples))

    assert num_m + num_s == total

    print 'Single accuracy = %s' %(np.sum(myAnswers[~multiples] == correctAnswers[~multiples]))
    print 'Multiple accuracy = %s' %(np.sum(myAnswers[multiples] == correctAnswers[multiples]))

    print 'All accuracy [%d]: %.4f' %(total, float(correct)/float(total))
    print 'Single accuracy [%d]: %.4f' %(num_s, np.sum(myAnswers[~multiples] == correctAnswers[~multiples])/num_s)
    print 'Multiple accuracy [%d]: %.4f' %(num_m, np.sum(myAnswers[multiples] == correctAnswers[multiples])/num_m)
        

## call different algorithms
if args.algo == 'slidingWindow':
    slidingWindow()
elif args.algo == 'linearRegression':
    linearRegression()
else:
    raise Exception(args.algo + " not implemented yet")

