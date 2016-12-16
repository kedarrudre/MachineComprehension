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
    def __init__(self, stopwords, args):
        self.rawPassage = None
        self.rawQuestions = []
        self.rawAnswers = []
        self.rawPassageLen = 0
        self.passage = []
        self.passageSentences = []
        self.passageAfterCoref = []
        self.passageSentencesAfterCoref = []
        self.questions = []
        self.answers = []
        self.author = None
        self.name = None
        self.time = None
        self.args = args
        self.params = defaultdict()
        self.params['noStopWordsInPassage'] = 1
        self.params['noStopWordsInQA'] = 1
        self.params['useLemma'] = 0
        self.params['lowercase'] = True ## fixme
        self.stopWords = stopwords
        self.correctAnswers = None
        self.tokenPositions = {}

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
        self.rawPassageLen = len(tokens)
        pos = 0
        for _ in tokens:
            __ = _.lower()
            tkn = self.caseConvert(_)
            if not self.params['noStopWordsInPassage']:
                self.passage.append(tkn)
            else:
                if __ not in self.stopWords:
                    self.passage.append(tkn)
            if tkn not in self.tokenPositions:
                self.tokenPositions[tkn] = [pos]
            else:
                self.tokenPositions[tkn].append(pos)
            pos += 1
        
        if self.params['useLemma']:
            tokens = nltk.pos_tag(tokens, tagset='universal')
            for _, tag in tokens:
                __ = _.lower()
                if not self.params['noStopWordsInPassage']:
                    self.passage.append(self.getLemmaWord((self.caseConvert(_), tag)))
                else:
                    if __ not in self.stopWords:
                        self.passage.append(self.getLemmaWord((self.caseConvert(_), tag)))

            for sent in sent_tokenize(story):
                tokens = nltk.word_tokenize(formatForProcessing(sent))
                tokens = nltk.pos_tag(tokens, tagset='universal')
                tokensAfterStopWords = []
                if self.params['noStopWordsInPassage']:
                    for _,tag in tokens:
                        __ = _.lower()
                        if __ not in self.stopWords:
                            tokensAfterStopWords.append(self.getLemmaWord((self.caseConvert(_),tag)))
                    else:
                        tokensAfterStopWords.append(self.getLemmaWord((self.caseConvert(_),tag)))
                
                self.passageSentences.append(tokensAfterStopWords)
        else:
            for _ in tokens:
                __ = _.lower()
                if not self.params['noStopWordsInPassage']:
                    self.passage.append(self.caseConvert(_))
                else:
                    if __ not in self.stopWords:
                        self.passage.append(self.caseConvert(_))

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

            ## If args.useCorefFeatures is True, then
            ## set few structures from story after coreference resolved
            if self.args.useCorefFeatures:
                fname = self.args.corefDataDir + '/'+ self.name + '.aftercoref.txt'
                if not os.path.isfile(fname):
                    print fname, " does not exist. Please generate stories after coreference resolved using gen_corenlp.py. See readme in data directory"
                    exit()
                f = open(fname, 'r')
                text = f.read().strip()
                tokens = nltk.word_tokenize(text)
                for _ in tokens:
                    __ = _.lower()
                    if not self.params['noStopWordsInPassage']:
                        self.passageAfterCoref.append(self.caseConvert(_))
                    else:
                        if __ not in self.stopWords:
                            self.passageAfterCoref.append(self.caseConvert(_))
                    
                for sent in sent_tokenize(text):
                    tokens = nltk.word_tokenize(sent)
                    tokensAfterStopWords = []
                    if self.params['noStopWordsInPassage']:
                        for _ in tokens:
                            __ = _.lower()
                            if __ not in self.stopWords:
                                tokensAfterStopWords.append(self.caseConvert(_))
                        else:
                            tokensAfterStopWords.append(self.caseConvert(_))
                
                    self.passageSentencesAfterCoref.append(tokensAfterStopWords)

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
    def __init__(self, fname, args):
        #fixme: check format of supported file
        self.stories = []
        self.args = args

        # collect stop words
        stopwordsSet = set()
        if args.useNltkStopWords:
            stopwordsSet = set(stopwords.words('english'))
        else:
            fin = open('/Users/msingh/cs221/project/mctDataSet/mctest-master/data/stopwords.txt','rU')
            fin = open('stopwords.txt','rU')
            for _ in fin:
                s = _.lower().strip()
                stopwordsSet.add(s)
            fin.close()

        # process stories
        fin = open(fname, 'rU')
        if args.verbose > 0:
            print 'Reading file %s: START' %(fname)
        for story in fin:
            story = story.strip()
            s = Story(stopwordsSet, self.args)
            data = re.split('\t', story)
            s.setName(data[0])
            if args.verbose > 4:
                print 'Reading story %s' %(data[0])
            properties = re.split(';', data[1])
            for p in properties:
                (name, v) = re.split(': ', p)
                if name == 'Author':
                    if args.verbose > 4:
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
        if args.verbose > 0:
            print 'Reading file %s: DONE' %(fname)

class MCTSlidingWindow(object):
    def __init__(self, args):
        self.icounts = defaultdict(lambda: 0.0)
        self.args = args
    
    def fit(self, stories):
        counts = defaultdict(lambda: 0)
        for story in stories:
            for token in story.passage:
                token = token.strip()
                counts[token] += 1.0
        if self.args.verbose > 4:
            print "counts %s" %(len(counts))
            for item, value in sorted(counts.items()):
                print item, value
            print "****************************"
                
        for token, token_count in counts.iteritems():
            self.icounts[token] = np.log(1.0 + 1.0/token_count)
        if self.args.verbose > 4:
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
        if self.args.verbose > 4:
            print 'passage is %s' %(passage)
            print 'passage at max_score (%s) is %s' %(max_score, passage_at_max_score)
        return max_score

    def predict(self, story):
        ANS_LETTERS = ['A', 'B', 'C', 'D']
        answers = []
        for id, (type, question) in enumerate(story.questions):
            question = set(question)
            if self.args.verbose > 3:
                print 'type = %s, question = %s, answers = %s' %(type, question, story.answers[id])

            scores = [self.predict_target(story.passage, question.union(set(choice))) \
                          for i, choice in enumerate(story.answers[id])]
            ans = ANS_LETTERS[scores.index(max(scores))]

            if self.args.verbose > 4:
                print 'scores: %s (%s)' %(scores, ans)

            answers.append(ans)
                
        return answers

class MCTLinearRegression(object):
    def __init__(self, args):
        self.args = args
        self.tfs = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.window_size = None
        # if False, then use setences for match.
        # if True, then use sliding window at every position of word in the passage
        self.slidingWindow = False
        self.clf = linear_model.SGDClassifier()
        self.X = []
        self.Y = []
        self.buckets = 20
        self.weights = defaultdict()
        self.icounts = defaultdict(lambda: 0.0)
        self.icountsPerStory = defaultdict(lambda: 0.0)
        self.cache = defaultdict()
        self.cache['features'] = {}
        self.features = []

        if args.sum_tfidf_on:
            self.features.append('sum_tfidf')
            
        if args.baseline_score_on:
            self.features.append('baseline_score')
            
        if args.sent_score_on:
            self.features.append('sent_score')

        if args.sent_perStory_score_on:
            self.features.append('sent_perStory_score')
            
        if args.sent2_perStory_score_on:
            self.features.append('sent2_perStory_score')

        if args.sent3_perStory_score_on:
            self.features.append('sent3_perStory_score')
        
        if args.question_negation_on:
            self.features.append('question_negation')

        if args.length:
            self.features.append('length_of_sentence')
            self.features.append('length_of_question')
            self.features.append('length_of_answer')

        if args.question_type_on:
            self.features.append('question is of type multiple')
            self.features.append('question is of type one')

        if args.useCorefFeatures:
            self.features.append('sent_perStory_score_after_coref')

#        for i in range(10):
#            self.features.append('length_of_common_words_in_qas is ' + str(i))

#        self.sum_tfidf_buckets = 5
#        self.sum_tfidf_max_score = 2.0

#        self.features.append('simlarity')

    def createBucketSumtfidf(self):
        bucketSize = self.sum_tfidf_max_score/self.sum_tfidf_buckets
        for i in range(self.sum_tfidf_buckets):
            self.features.append('sum_tfidf is between '+str(bucketSize*i)+' and '+str(bucketSize*(i+1)))
            
        self.features.append('sum_tfidf is more than ' + str(self.sum_tfidf_max_score))

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
            if self.args.verbose > 4:
                print 'question is %s' %(q)
            for ano, a in enumerate(story.answers[qno]):
                scores = []
                minScore = float('inf')
                maxScore = float('-inf')
                if self.args.verbose > 4:
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
                if self.args.verbose > 4:
                    print scores
                    print minScore
                    print maxScore
                if step != 0:
                    for x in scores:
                        s = x - minScore
                        if self.args.verbose > 4:
                            print int(s/step)
                        features[int(s/step)-1] = 1

                X.append(features)
                if self.args.verbose > 4:
                    print story.correctAnswers
                Y.append(story.correctAnswers[qno] == ano)
            
        return (X,Y)

    def extractFeatures(self, story, sid, qid, aid):
        features = defaultdict(float)
        # sum of tfidf of common words in sentence and (q+a)
        sum_tfidf = 0.0
        # sliding window bag of word as in baseline
        baseline_score = 0.0
        # sliding window bag of word as in baseline (consider one sentence)
        sent_score = 0.0
        # sliding window bag of word as in baseline (consider one sentence, and iCounts is per story)
        sent_perStory_score = 0.0
        # sliding window bag of word as in baseline (consider two sentence, and iCounts is per story)
        sent2_perStory_score = 0.0
        # sliding window bag of word as in baseline (consider three sentence, and iCounts is per story)
        sent3_perStory_score = 0.0

        # after coreference resolved
        sent_perStory_score_after_coref = 0.0

        if (story.name, sid, qid, aid) in self.cache['features']:
            return self.cache['features'][(story.name, sid, qid, aid)]
        
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
        number_of_common_words = 0
        for word in story.passageSentences[sid]:
            if word in target:
                number_of_common_words += 1
                sum_tfidf += get_tfidf(word, response, feature_names)
                #print 'adding tfidf %s of %s' %(get_tfidf(word, response, feature_names), word)
        if self.args.sum_tfidf_on:
            features['sum_tfidf'] = sum_tfidf

        # buckets of sum_tfidf
#         sum_tfidf_bucketSize = self.sum_tfidf_max_score/self.sum_tfidf_buckets
#         sum_tfidf_bucket = int(sum_tfidf/sum_tfidf_bucketSize)
#         if sum_tfidf_bucket >= self.sum_tfidf_max_score:
#             features['sum_tfidf is more than ' + str(self.sum_tfidf_max_score)] = 1
#         else:
#             features['sum_tfidf is between '+str(sum_tfidf_bucket*sum_tfidf_bucketSize)+' and '+str(sum_tfidf_bucketSize*(sum_tfidf_bucket+1))] = 1

        #baseline_score
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

        if self.args.baseline_score_on:
            features['baseline_score'] = baseline_score
            
        # baseline_sentence_score and baseline_sentence_perStory_score
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
            if score > sent_score:
                sent_score = score
            if scorePerStory > sent_perStory_score:
                sent_perStory_score = scorePerStory

        if self.args.sent_score_on:
            features['sent_score'] = sent_score

        if self.args.sent_perStory_score_on:
            features['sent_perStory_score'] = sent_perStory_score

        # score over two sentences
        allSentences = []
        if sid+1 < len(story.passageSentences):
            allSentences = story.passageSentences[sid:sid+1]

        for sent in allSentences:
            for i in xrange(len(sent)):
                score = 0.0
                for j in xrange(window_size):
                    if i+j < len(sent) and sent[i+j] in target:
                        score += self.icountsPerStory[story.name][sent[i+j]]
                if score > sent2_perStory_score:
                    sent2_perStory_score = score

        if self.args.sent2_perStory_score_on:
            features['sent2_perStory_score'] = sent2_perStory_score

        # score over three sentences
        allSentences = []
        if sid+2 < len(story.passageSentences):
            allSentences = story.passageSentences[sid:sid+2]
            
        for sent in allSentences:
            for i in xrange(len(sent)):
                score = 0.0
                for j in xrange(window_size):
                    if i+j < len(sent) and sent[i+j] in target:
                        score += self.icountsPerStory[story.name][sent[i+j]]
                if score > sent3_perStory_score:
                    sent3_perStory_score = score
         
        if self.args.sent3_perStory_score_on:
            features['sent3_perStory_score'] = sent3_perStory_score
        

        # question_negation
        # quesion contains ^wh.*'n't or not'
        if self.args.question_negation_on:
            if re.search('^wh.*(n\'t|not)', story.rawQuestions[qid][1]):
                features['question_negation'] = 1

        # length features
        if self.args.length and story.questions[qid][0] == 'multiple':
            features['length_of_sentence'] = len(story.passageSentences[sid])
            features['length_of_question'] = len(story.questions[qid][1])
            features['length_of_answer'] = len(story.answers[qid][aid])
            features['length_of_common_words_in_qas is ' + str(number_of_common_words)] = 1.0
        
        # type of question
        if self.args.question_type_on:
            features['question is of type ' +  str(story.questions[qid][0])] = 1.0

        # after coreference features
        if self.args.useCorefFeatures:
            if sid < len(story.passageSentencesAfterCoref):
                sentence = story.passageSentencesAfterCoref[sid]
            else:
                sentence = story.passageSentencesAfterCoref[-1]
            window_size = len(target)
            sentence_length = len(sentence)
            for i in xrange(sentence_length):
                score = 0.0
                for j in xrange(window_size):
                    if i+j < sentence_length and sentence[i+j] in target:
                        score += self.icountsPerStory[story.name][sentence[i+j]]
                if score > sent_perStory_score_after_coref:
                    sent_perStory_score_after_coref = score

            if story.questions[qid][0] == 'multiple':
                features['sent_perStory_score_after_coref'] = sent_perStory_score_after_coref

        # cosine simlarity between sentence and q,a
#         a = defaultdict(lambda: 0.0)
#         b = defaultdict(lambda: 0.0)
#         if sid < len(story.passageSentencesAfterCoref):
#             for word in story.passageSentencesAfterCoref[sid]:
#                 a[word] = get_tfidf(word, response, feature_names)

#         for word in story.answers[qid][aid]:
#             b[word] = get_tfidf(word, response, feature_names)
            
#        features['simlarity'] = util.cosineSimlarity(a,b)
       
        #DistanceBased feature
        Q_set = set(q)
        Ai_set = set(a)
        PW_set = set(sentence)
        SQ_set = PW_set.intersection(Q_set) # Stopwords are already removed?
        SAi_set = PW_set.intersection(Ai_set).difference(Q_set) # Words present in Passage and Answer, but not in Question.
        feature_value = 0

        if len(SQ_set) == 0 or len(SAi_set) == 0:
            feature_value = 1
        else:
            min = 100000 #TODO: Fix to int max.
            for q_tkn in  Q_set:
                for a_tkn in Ai_set:
                    if q_tkn not in story.tokenPositions or a_tkn not in story.tokenPositions:
                        continue
                    for q_tkn_pos in story.tokenPositions[q_tkn]:
                        for a_tkn_pos in story.tokenPositions[a_tkn]:
                            if abs(q_tkn_pos - a_tkn_pos) < min:
                                min = abs(q_tkn_pos - a_tkn_pos)
            feature_value = float((min + 1))/(story.rawPassageLen - 1)
        features['distance_based'] = feature_value

        self.cache['features'][(story.name, sid, qid, aid)] = features
        
        return features
    
    def fit_OLD(self, stories):
        token_dict = {}
        for story in stories:
            token_dict[story.name] = story.rawPassage
        self.tfs = self.tfidf.fit_transform(token_dict.values())
        if self.args.verbose > 4:
            print self.tfidf.get_feature_names()

        for story in stories:
            if self.args.verbose > 4:
                print story.correctAnswers
                print story.passage
                print story.questions
                print story.answers
            X, Y = self.extractFeatures(story)
            self.X += X
            self.Y += Y

        # fit 
        if self.args.verbose > 3:
            print 'fitting'
        self.clf.fit(np.array(self.X), np.array(self.Y))
        
    def fit(self, stories):
        token_dict = {}
        for story in stories:
            token_dict[story.name] = story.rawPassage
        self.tfs = self.tfidf.fit_transform(token_dict.values())
        if self.args.verbose > 4:
            print self.tfidf.get_feature_names()

        # create buckets of sum_tfidf
#        self.createBucketSumtfidf()
        self.weights = util.cccp(self.extractFeatures, stories, self.args, self.features)

    def calc_icounts(self, stories):
        counts = defaultdict(lambda: 0)
        countsPerStory = defaultdict(lambda: 0)
        for story in stories:
            countsPerStory[story.name] = defaultdict(lambda: 0)
            for token in story.passage:
                token = token.strip()
                counts[token] += 1.0
                countsPerStory[story.name][token] += 1
        if self.args.verbose > 4:
            print "counts %s" %(len(counts))
            for item, value in sorted(counts.items()):
                print item, value
            print "****************************"
                
        for token, token_count in counts.iteritems():
            self.icounts[token] = np.log(1.0 + 1.0/token_count)

        for story in stories:
            self.icountsPerStory[story.name] = defaultdict(lambda: 0.0)
            for tkn, tknCnt in countsPerStory[story.name].iteritems():
                self.icountsPerStory[story.name][tkn] = np.log(1.0 + 1.0/tknCnt)

        if self.args.verbose > 4:
            print "icounts are:"
            for item, value in sorted(self.icounts.iteritems()):
                print item, value

    # predict
    def predict_OLD(self, stories):
        if self.args.verbose > 3:
            print 'Predicting'
        ans = []
        for story in stories:
            if self.args.verbose > -1:
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
        if self.args.verbose > 3:
            print 'Predicting'
        
        if self.args.verbose:
            print "weights are ", self.weights

        ANS_LETTERS = ['A', 'B', 'C', 'D']
        ans = []
        for story in stories:
            if self.args.verbose > 0:
                print story.name
                print formatForPrint(story.rawPassage), "\n"
                print story.rawQuestions,  "\n"
                print story.rawAnswers, "\n"

            for qid in range(len(story.questions)):
                scores = []
                for aid in range(len(story.answers[qid])):
                    score = max([(util.dotProduct(self.weights, self.extractFeatures(story, sid, qid, aid)),sid) for \
                                     sid in range(len(story.passageSentences))])
                    scores.append((score, aid))
                # if question contains "n't | not", and begin
                # with "what, who, whose", select the minium score.
                s = story.rawQuestions[qid][1].strip()
                if re.search('^(who|what|whose).*(n\'t|not)', s, flags=re.IGNORECASE):
                    answer = min(scores)[1]
                else:
                    answer = max(scores)[1]

                ans.append(ANS_LETTERS[answer])

                if self.args.verbose > 0:
                    if answer != story.correctAnswers[qid]:
                        print 'WRONG: %s: correct answer %s, predicted answer %s, scores %s' \
                            %(story.rawQuestions[qid][0], ANS_LETTERS[story.correctAnswers[qid]], ANS_LETTERS[answer], scores)
                    else:
                        print 'RIGHT: %s: correct answer %s, predicted answer %s, scores %s' \
                            %(story.rawQuestions[qid][0], ANS_LETTERS[story.correctAnswers[qid]], ANS_LETTERS[answer], scores)

            if self.args.verbose > 0:
                print "\n"
                        
        return ans
