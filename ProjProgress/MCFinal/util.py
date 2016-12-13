#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import os
import sys
from collections import defaultdict
import numpy as np
import re
import random
import copy
import pickle
from shutil import copyfile

# Common useful functions
def formatForPrint(str):
    s = re.sub(r"\\newline", "\\n", str)
    return s

def formatForProcessing(str):
    s = re.sub(r"\\newline", " ", str)
    s = re.sub(r"\'s", "", s)
    return s

def answerLetters2Num(x):
    return {'A':0, 'B':1, 'C':2, 'D':3}[x]

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def cccp(f, stories, params):
    # f is feature extractor function f(story, sid, qid, aid) where
    # story -> story object
    # sid -> sentence id or window id
    # qid -> question id
    # aid -> answer id
    # returns feature vector as sparse dictionary

    # stories is a list that contains list of story objects
    # T number of iterations to run
    # returns weights
    weights = defaultdict(float)
    if params['startFromExistingWeights']:
        if os.path.isfile('weights.txt'):
            with open('weights.txt', 'rb') as myWeightsFile:
                weights = pickle.loads(myWeightsFile.read())
                myWeightsFile.close()
                copyfile('weights.txt', 'weights.prev.txt')
                print 'Reading existing weights to start from weights.txt'
        else:
            exit('weights.txt file is not found')
    else:
        for item in params['features']:
            weights[item] = random.random()

    if params['read_weights_from_file']:
        if os.path.isfile('weights.txt'):
            with open('weights.txt', 'rb') as myWeightsFile:
                weights = pickle.loads(myWeightsFile.read())
                myWeightsFile.close()
                copyfile('weights.txt', 'weights.prev.txt')
                print 'Reading existing weights'
        return weights

    optimalSentenceIds = []
    hyperParamEta = params['eta']
    hyperParamT = params['T']
    hyperParamLambda = params['lambda']
    hyperParamDelta = params['delta']
    
    def func():
        print 'weights are %s' %(weights)
        ans = hyperParamLambda * dotProduct(weights, weights)
        for id, story in enumerate(stories):
            for qid in range(len(story.questions)):
                ans -= max([dotProduct(weights, f(story, w, qid, story.correctAnswers[qid])) for \
                                w in range(len(story.passageSentences))])
                ans += max([max([dotProduct(weights, f(story, w, qid, aid)) for \
                                     w in range(len(story.passageSentences))]) + story.correctAnswers[qid] != aid for \
                                aid in range(len(story.answers))])
        return ans

    def func1(story, qid, weights_copy):
        ans = hyperParamLambda * dotProduct(weights_copy, weights_copy)
        ans -= max([dotProduct(weights_copy, f(story, w, qid, story.correctAnswers[qid])) for \
                        w in range(len(story.passageSentences))])
        ans += max([max([dotProduct(weights_copy, f(story, w, qid, aid)) for \
                             w in range(len(story.passageSentences))]) + story.correctAnswers[qid] != aid for \
                        aid in range(len(story.answers))])
        return ans

    def stochasticGradientDescent():
        def max_over_a(story, id, qid):
            return max([(dotProduct(weights, f(story, optimalSentenceIds[id][qid], qid, aid)), aid) for \
                            aid in range(len(story.answers[qid]))])

        def df(w, story, id, qid):
            # calculate numerically
            # f(w+delta) - f(w-delta)/(2*delta)
            d1 = defaultdict(float)
            for item in weights:
                w1 = copy.deepcopy(weights)
                w2 = copy.deepcopy(weights)
                w1[item] += hyperParamDelta
                w2[item] -= hyperParamDelta
                d1[item] = (func1(story, qid, w1) - func1(story, qid, w2))/(2*hyperParamDelta)
            return d1
            

        def df_old(w, story, id, qid):
            d1 = defaultdict(float)
            optimalAnswer = max_over_a(story, id, qid)[1]
            prediction = f(story, optimalSentenceIds[id][qid], qid, optimalAnswer)
            target = f(story, optimalSentenceIds[id][qid], qid, story.correctAnswers[qid])
            increment(d1, hyperParamLambda, w)
            increment(d1, 1, target)
            increment(d1, -1, prediction)
#            print 'max_over_a answer %s, correctAnswer = %s' %(optimalAnswer, story.correctAnswers[qid])
            return d1
    
        for itr in range(hyperParamT):
            for id, story in enumerate(stories):
                for qid in range(len(story.questions)):
                    gradient = df(weights, story, id, qid)
#                    print 'gradient is %s, eta*gradient = %s' %(gradient, hyperParamEta*gradient['sum_tfidf'])
                    increment(weights, -hyperParamEta, gradient)
                print 'weight is %s' %(weights)

    for itr in range(params['number_of_cccp_iterations']):
        # step1, fixed weights, compute optimum w that minimizes
        # max_{w \in W} w.f(Pi, w, qi, ai)
        for id, story in enumerate(stories):
            wm = []
            for qid in range(len(story.questions)):
                scores = [(dotProduct(weights, f(story, w, qid, story.correctAnswers[qid])),w) for \
                              w,_ in enumerate(story.passageSentences)]
                a = max(scores)
#                a = max([(dotProduct(weights, f(story, w, qid, story.correctAnswers[qid])),w) for \
#                             w,_ in enumerate(story.passageSentences)])
#                print 'question is %s, answer = %s ' %(story.questions[qid], story.answers[qid])
#                print 'story is %s, qid = %s, correctAnswer=%s, scores = %s' %(story.name, qid, story.correctAnswers[qid], scores)
#                print a
                wm.append(a[1])
            optimalSentenceIds.append(wm)

        print optimalSentenceIds

        # step 2
        stochasticGradientDescent()
        
        # print the loss
        print('iteration %s, f = %s' %(itr, func()))

        # empty optimalSentenceIds
        optimalSentenceIds = []
        
    if os.path.isfile('weights.txt'):  
        copyfile('weights.txt', 'weights.1.txt')
    with open('weights.txt', 'wb') as myWeightsFile:
        pickle.dump(weights, myWeightsFile)
        myWeightsFile.close()
    
    return weights




