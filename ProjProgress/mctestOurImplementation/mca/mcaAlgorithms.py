#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import os
import sys
from collections import defaultdict
import numpy as np
import re
from util import formatForPrint, formatForProcessing
from nltk.corpus import stopwords

class Story(object):
    def __init__(self, stopwords):
        self.rawPassage = None
        self.rawQuestions = []
        self.rawAnswers = []
        self.passage = []
        self.questions = []
        self.answers = []
        self.author = None
        self.name = None
        self.time = None
        self.params = defaultdict()
        self.params['noStopWords'] = 1
        self.stopWords = stopwords

    def updateParameter(self, p, v):
        self.params[p] = v

    def setAuthor(self, x):
        self.author = x
        
    def setName(self, x):
        self.name = x
        
    def setTime(self, x):
        self.time = x
        
    def setStory(self, story):
        self.rawPassage = story
        tokens = nltk.word_tokenize(formatForProcessing(story))
        if self.params['noStopWords']:
            for _ in tokens:
                __ = _.lower()
                if __ not in self.stopWords:
                    self.passage.append(_)
        else:
            self.passage = tokens
        
    def setQuestion(self, q, answers):
        (flag, question) = re.split(':', q)
        self.rawQuestions.append((flag, question))
        
        tokens = nltk.word_tokenize(question)
        questionTokens = []
        for _ in tokens:
            if _ == '?':
                continue
            if self.params['noStopWords']:
                __ = _.lower()
                if __ not in self.stopWords:
                    questionTokens.append(_)
            else:
                questionTokens.append(_)
        self.questions.append((flag, set(questionTokens)))

        answerList = []
        rawAnswerList = []

        for answer in answers:
            tokens = nltk.word_tokenize(answer)
            answerTokens = []
            for _ in tokens:
                if self.params['noStopWords']:
                    __ = _.lower()
                    if __ not in self.stopWords:
                        answerTokens.append(_)
                else:
                    answerTokens.append(_)
            answerList.append(set(answerTokens))
            rawAnswerList.append(answer)

        self.answers.append(answerList)
        self.rawAnswers.append(rawAnswerList)

class MCTReadData(object):
    def __init__(self, fname, verbose):
        #fixme: check format of supported file
        self.stories = []

        # collect stop words
        stopwords = set()
        fin = open('data/stopwords.txt','rU')
        for stopword in fin:
            s = stopword.lower().strip()
            stopwords.add(s)
        fin.close()

        # process stories
        fin = open(fname, 'rU')
        if verbose > 0:
            print 'Reading file %s: START' %(fname)
        for story in fin:
            story = story.strip()
            s = Story(stopwords)
            data = re.split('\t', story)
            s.setName(data[0])
            if verbose > 4:
                print 'Reading story %s' %(data[0])
            properties = re.split(';', data[1])
            for p in properties:
                (name, v) = re.split(': ', p)
                if name == 'Author':
                    if verbose > 4:
                        print 'Setting Author %s for story %s' %(v, data[0])
                    s.setAuthor(v)
                    continue
                if name == 'Work Time(s)' or name == 'Work Time':
                    s.setTime(v)
                continue
            s.setStory(data[2])
            index = 3
            while True:
                s.setQuestion(data[index], data[index+1:index+5])
                index += 5
                if index + 5 > len(data):
                    break
            self.stories.append(s)
        fin.close()
        if verbose > 0:
            print 'Reading file %s: DONE' %(fname)

class MCTSlidingWindow(object):
    def __init__(self, verbose):
        self.icounts = {}
        self.verbose = verbose
    
    def fit(self, stories):
        counts = defaultdict(lambda: 0)
        for story in stories:
            for token in story.passage:
                token = token.strip()
                counts[token] += 1.0
        if self.verbose > 4:
            print "counts %s" %(len(counts))
            for item, value in sorted(counts.items()):
                print item, value
            print "****************************"
                
        for token, token_count in counts.iteritems():
            self.icounts[token] = np.log(1.0 + 1.0/token_count)
        if self.verbose > 4:
            print "icounts are:"
            for item, value in sorted(self.icounts.iteritems()):
                print item, value

    def predict_target(self, passage, target):
        max_score = 0.0
        window_size = len(target)
        passage_length = len(passage)
        for i in xrange(passage_length):
            score = 0.0
            for j in xrange(window_size):
                if i+j < passage_length:
                    if passage[i+j] in target:
                        score += self.icounts[passage[i+j]]
            if score > max_score:
                max_score = score
        return max_score

    def predict(self, story, file):
        ANS_LETTERS = ['A', 'B', 'C', 'D']
        answers = []
        allScores = []
        for id, (type, question) in enumerate(story.questions):
            if self.verbose > 3:
                print 'type = %s, question = %s, answers = %s' %(type, question, story.answers[id])

            scores = [self.predict_target(story.passage, question.union(choice)) \
                          for i, choice in enumerate(story.answers[id])]
            ans = ANS_LETTERS[scores.index(max(scores))]

            #if self.verbose > 4:
            print 'scores: [%s] %s (%s)' %(story.name, scores, ans)
            
            for score in scores:
                # file.write(story.name + " : " + str(score) + "\n")
                #file.write(str(score) + "\n")
                allScores.append(score)

            answers.append(ans)
        return answers, allScores
