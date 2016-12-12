#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import nltk
import os
import sys
from collections import defaultdict
import numpy as np
import re
from util import formatForPrint, formatForProcessing
import util
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

lemmatizer = nltk.WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('A'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class Story(object):
    def __init__(self, stopwords):
        self.rawPassage = None
        self.rawQuestions = []
        self.rawAnswers = []
        self.passage = []
        self.passageSentences = []
        self.questions = []
        self.answers = []
        self.author = None
        self.name = None
        self.time = None
        self.params = defaultdict()
        self.params['noStopWordsInPassage'] = 1
        self.params['noStopWordsInQA'] = 1
        self.params['useLemma'] = 0
        self.params['lowercase'] = True ## fixme
        self.stopWords = stopwords
        self.correctAnswers = None

    def updateParameter(self, p, v):
        self.params[p] = v

    def setAuthor(self, x):
        self.author = x
        
    def setName(self, x):
        self.name = x
        
    def setTime(self, x):
        self.time = x

    def getLemmaWord(self, x):
        return lemmatizer.lemmatize(x[0], get_wordnet_pos(x[1]))

    def caseConvert(self, x):
        if self.params['lowercase']:
            return x.lower()
        return x

    def setStory(self, story):
        self.rawPassage = story
        tokens = nltk.word_tokenize(formatForProcessing(story))
        for _ in tokens:
            __ = _.lower()
            if not self.params['noStopWordsInPassage']:
                self.passage.append(self.caseConvert(_))
            else:
                if __ not in self.stopWords:
                    self.passage.append(self.caseConvert(_))
        
        if self.params['useLemma']:
            self.passage = nltk.pos_tag(self.passage, tagset='universal')
            t = []
            for _ in self.passage:
                t.append(self.getLemmaWord(_))
            self.passage = t

        for sent in sent_tokenize(story):
            tokens = nltk.word_tokenize(formatForProcessing(sent))
            tokensAfterStopWords = []
            if self.params['noStopWordsInPassage']:
                for _ in tokens:
                    __ = _.lower()
                    if __ not in self.stopWords:
                        tokensAfterStopWords.append(self.caseConvert(_))
            else:
                tokensAfterStopWords.append(self.caseConvert(_))
            self.passageSentences.append(tokensAfterStopWords)
        

    def setCorrectAnswer(self, x):
        self.correctAnswers = x
        
    def setQuestion(self, q, answers):
        (flag, question) = re.split(':', q)
        self.rawQuestions.append((flag, question))
        
        tokens = nltk.word_tokenize(question)
        questionTokens = []
        for _ in tokens:
            if _ == '?':
                continue
            if self.params['noStopWordsInQA']:
                __ = _.lower()
                if __ not in self.stopWords:
                    questionTokens.append(self.caseConvert(_))
            else:
                questionTokens.append(self.caseConvert(_))
                
        if self.params['useLemma']:
            questionTokens = nltk.pos_tag(questionTokens, tagset='universal')
            t = []
            for _ in questionTokens:
                t.append(self.getLemmaWord(_))
            questionTokens = t
        self.questions.append((flag, questionTokens))

        answerList = []
        rawAnswerList = []

        for answer in answers:
            tokens = nltk.word_tokenize(answer)
            answerTokens = []
            for _ in tokens:
                if self.params['noStopWordsInQA']:
                    __ = _.lower()
                    if __ not in self.stopWords:
                        answerTokens.append(self.caseConvert(_))
                else:
                    answerTokens.append(self.caseConvert(_))
            
            if self.params['useLemma']:
                answerTokens = nltk.pos_tag(answerTokens, tagset='universal')
                t = []
                for _ in answerTokens:
                    t.append(self.getLemmaWord(_))
                answerTokens = t

            answerList.append(answerTokens)
            rawAnswerList.append(answer)

        self.answers.append(answerList)
        self.rawAnswers.append(rawAnswerList)

class MCTReadData(object):
    def __init__(self, fname, verbose):
        #fixme: check format of supported file
        self.stories = []

        # collect stop words
        stopwords = set()
        fin = open('C:\Personal\Learnings\Stanford\Project\src\MachineComprehension\ProjProgress\MCFinal\data\stopwords.txt','rU')
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
        passage_at_max_score = None
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
                passage_at_max_score = passage[i:i+j]
        if self.verbose > 4:
            print 'passage is %s' %(passage)
            print 'passage at max_score (%s) is %s' %(max_score, passage_at_max_score)
        return max_score

    def predict(self, story):
        ANS_LETTERS = ['A', 'B', 'C', 'D']
        answers = []
        for id, (type, question) in enumerate(story.questions):
            question = set(question)
            if self.verbose > 3:
                print 'type = %s, question = %s, answers = %s' %(type, question, story.answers[id])

            scores = [self.predict_target(story.passage, question.union(set(choice))) \
                          for i, choice in enumerate(story.answers[id])]
            ans = ANS_LETTERS[scores.index(max(scores))]

            if self.verbose > 4:
                print 'scores: %s (%s)' %(scores, ans)

            answers.append(ans)
                
        return answers

class MCTLinearRegression(object):
    def __init__(self, verbose):
        self.verbose = verbose
        self.tfs = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.window_size = None
        # if False, then use setences for match.
        # if True, then use sliding window at every position of word in the passage
        self.slidingWindow = True
        self.clf = linear_model.SGDClassifier()
        self.X = []
        self.Y = []
        self.buckets = 20
        self.weights = defaultdict()
        self.icounts = {}
        self.icountsPerStory = {}
        self.cache = defaultdict()
        self.params = {'read_weights_from_file': 0,
                       'number_of_cccp_iterations': 2,
                       'eta': 0.005,
                       'T': 1,
                       'lambda': 0.1,
                       'delta': 0.001,
                       'features': ['sum_tfidf','baseline_score', 'baseline_sentence_score', 'baseline_sentence_perStory_score']
                       }

    def updateParameter(self, p, v):
        self.params[p] = v

    def extractFeatures_OLD(self, story):
        # extract features
        # sum of tf-idf of words that occur in question + answer
        # and in a sentence in story
        X = []
        Y = []
        def get_tfidf(name, response, feature_names):
            if name in feature_names:
                for col in response.nonzero()[1]:
                    if feature_names[col] == name:
                        return response[0, col]
            return 0.0

        passage_length = len(story.passage)
        feature_names = self.tfidf.get_feature_names()
        response = self.tfidf.transform([story.rawPassage])

#        for col in response.nonzero()[1]:
#            print feature_names[col], ' - ', response[0, col]

        for qno, typeAndq in enumerate(story.questions):
            type, q = typeAndq
            if self.verbose > 4:
                print 'question is %s' %(q)
            for ano, a in enumerate(story.answers[qno]):
                scores = []
                minScore = float('inf')
                maxScore = float('-inf')
                if self.verbose > 4:
                    print 'answer is %s' %(a)
                target = q.union(a)
                if self.slidingWindow:
                    if self.window_size is None:
                        window_size = len(target)
                    for i in xrange(passage_length):
                        score = 0.0
                        for j in xrange(window_size):
                            if i+j < passage_length:
                                if story.passage[i+j] in target:
                                    score += get_tfidf(story.passage[i+j], response, feature_names)
                        scores.append(score)
                        if score < minScore:
                            minScore = score
                        if score > maxScore:
                            maxScore = score
                else:
                    for i, sent in enumerate(story.passageSentences):
                        score = 0.0
                        for word in sent:
                            if word in target:
                                score += get_tfidf(word, response, feature_names)
                        scores.append(score)
                        if score < minScore:
                            minScore = score
                        if score > maxScore:
                            maxScore = score
                    
                # bucketize scores
                scoreRange = maxScore - minScore
                step = scoreRange/self.buckets
                features = [0] * self.buckets
                if self.verbose > 4:
                    print scores
                    print minScore
                    print maxScore
                if step != 0:
                    for x in scores:
                        s = x - minScore
                        if self.verbose > 4:
                            print int(s/step)
                        features[int(s/step)-1] = 1

                X.append(features)
                if self.verbose > 4:
                    print story.correctAnswers
                Y.append(story.correctAnswers[qno] == ano)
            
        return (X,Y)

    def extractFeatures(self, story, sid, qid, aid):
        features = defaultdict()
        # sum of tfidf of common words in sentence and (q+a)
        sum_tfidf = 0.0
        # sliding window bag of word as in baseline (consider entire passage)
        baseline_score = 0.0
        # sliding window bag of word as in baseline (consider one sentence)
        baseline_sentence_score = 0.0
        # sliding window bag of word as in baseline (consider one sentence, and iCounts is per story)
        baseline_sentence_perStory_score = 0.0
        
        def get_tfidf(name, response, feature_names):
            if name in feature_names:
                for col in response.nonzero()[1]:
                    if feature_names[col] == name:
                        return response[0, col]
            return 0.0

        passage_length = len(story.passage)
        feature_names = self.tfidf.get_feature_names()
        response = self.tfidf.transform([story.rawPassage])
        type, q = story.questions[qid]
        a = story.answers[qid][aid]
        target = set(q).union(set(a))

        # sum_tfidf
        for word in story.passageSentences[sid]:
            if word in target:
                sum_tfidf += get_tfidf(word, response, feature_names)
                #print 'adding tfidf %s of %s' %(get_tfidf(word, response, feature_names), word)
        features['sum_tfidf'] = sum_tfidf

        # baseline_score
        try:
            if self.cache['baseLineScore_story_name'] == story.name and \
                    self.cache['baseLineScore_qid'] == qid and \
                    self.cache['baseLineScore_aid'] == aid:
                baseline_score = self.cache['baseLineScore_score']
        except:
            passage = story.passage
            window_size = len(target)
            passage_length = len(passage)
            for i in xrange(passage_length):
                score = 0.0
                for j in xrange(window_size):
                    if i+j < passage_length:
                        if passage[i+j] in target:
                            score += self.icounts[passage[i+j]]
                if score > baseline_score:
                    baseline_score = score
            self.cache['baseLineScore_story_name'] = story.name
            self.cache['baseLineScore_qid'] = qid
            self.cache['baseLineScore_aid'] = aid
            self.cache['baseLineScore_score'] = baseline_score

        features['baseline_score'] = baseline_score
        
        # baseline_sentence_score
        sentence = story.passageSentences[sid]
        window_size = len(target)
        sentence_length = len(sentence)
        for i in xrange(sentence_length):
            score = 0.0
            scorePerStory = 0.0
            for j in xrange(window_size):
                if i+j < sentence_length and sentence[i+j] in target:
                    score += self.icounts[sentence[i+j]]
                    scorePerStory += self.icountsPerStory[story.name][sentence[i+j]]
            if score > baseline_sentence_score:
                baseline_sentence_score = score
            if scorePerStory > baseline_sentence_perStory_score:
                baseline_sentence_perStory_score = scorePerStory
        features['baseline_sentence_score'] = baseline_sentence_score
        features['baseline_sentence_perStory_score'] = baseline_sentence_perStory_score

        return features
        
    def fit_OLD(self, stories):
        token_dict = {}
        for story in stories:
            token_dict[story.name] = story.rawPassage
        self.tfs = self.tfidf.fit_transform(token_dict.values())
        if self.verbose > 4:
            print self.tfidf.get_feature_names()

        for story in stories:
            if self.verbose > 4:
                print story.correctAnswers
                print story.passage
                print story.questions
                print story.answers
            X, Y = self.extractFeatures(story)
            self.X += X
            self.Y += Y

        # fit 
        if self.verbose > 3:
            print 'fitting'
        self.clf.fit(np.array(self.X), np.array(self.Y))
        
    def fit(self, stories):
        token_dict = {}
        for story in stories:
            token_dict[story.name] = story.rawPassage
        self.tfs = self.tfidf.fit_transform(token_dict.values())
        if self.verbose > 4:
            print self.tfidf.get_feature_names()

        self.weights = util.cccp(self.extractFeatures, stories, self.params)

    def calc_icounts(self, stories):
        counts = defaultdict(lambda: 0)
        countsPerStory = defaultdict(lambda: 0)
        for story in stories:
            countsPerStory[story.name] = defaultdict(lambda: 0)
            for token in story.passage:
                token = token.strip()
                counts[token] += 1.0
                countsPerStory[story.name][token] += 1
        if self.verbose > 4:
            print "counts %s" %(len(counts))
            for item, value in sorted(counts.items()):
                print item, value
            print "****************************"
                
        for token, token_count in counts.iteritems():
            self.icounts[token] = np.log(1.0 + 1.0/token_count)
        
        for story in stories:
            self.icountsPerStory[story.name] = defaultdict()
            for tkn, tknCnt in countsPerStory[story.name].iteritems():
                self.icountsPerStory[story.name][tkn] = np.log(1.0 + 1.0/tknCnt)

        if self.verbose > 4:
            print "icounts are:"
            for item, value in sorted(self.icounts.iteritems()):
                print item, value

    # predict
    def predict_OLD(self, stories):
        if self.verbose > 3:
            print 'Predicting'
        ans = []
        for story in stories:
            if self.verbose > -1:
                print story.correctAnswers
                print story.passage
                print story.questions
                print story.answers
            X, _ = self.extractFeatures(story)
            Y = self.clf.predict(X)
            # 4 possible answers for every question
            i = 0
            while i < len(Y):
                choices = []
                for _ in Y[i:i+4]:
                    if _ is True:
                        choices.append(['A','B','C','D'][i%4])
                if len(choices) > 0:
                    x = random.choice(choices)
                    print 'Selected %s from len(choices) valid answers' %(x)

                else:
                    x = random.choice(['A','B','C','D'])
                    print 'Selected random %s' %(x)
                ans.append(x)
                i = i+4
                
        return ans

    def predict(self, stories):
        if self.verbose > 3:
            print 'Predicting'

        ANS_LETTERS = ['A', 'B', 'C', 'D']
        ans = []
        for story in stories:
            if self.verbose > -1:
                print story.correctAnswers
                print story.passage
                print story.questions
                print story.answers

                for qid in range(len(story.questions)):
                    answer = max([(max([(util.dotProduct(self.weights, self.extractFeatures(story, sid, qid, aid)),sid) for \
                                            sid in range(len(story.passageSentences))])[0], aid) for \
                                      aid in range(len(story.answers[qid]))])[1]
                    ans.append(ANS_LETTERS[answer])

        return ans
