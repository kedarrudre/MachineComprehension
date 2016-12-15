#!/usr/bin/python

import re
import nltk
import copy
import sys
import os

cmd = 'java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref'
corenlpDir = 'stanford-corenlp-full-2016-10-31'
outDir = 'mctDataSetAfterCoref'

def formatForPrint(str):
    s = re.sub(r"\\newline", "\\n", str)
    return s

def generate_story(fileNameIn, fileNameOut):

    fin = open(fileNameIn, 'r')
    fout = open(fileNameOut, 'w')

    sentences = []
    newSentences = []
    inSentence = 0
    inCoreference = 0
    coreferenceStart = False

    for line in fin.readlines():
        line = line.strip('\n')
        if inSentence:
            if re.match('^\[Text\=', line):
                sentences.append(sent)
                inSentence = 0
            else:
                if sent != "":
                    sent = sent + " " + line
                else:
                    sent = line
            
        if re.match('^Sentence ', line):
            inSentence = 1
            sent = ""
            continue
    
        if re.match('^Coreference set', line):
            inCoreference = 1
            continue
    
        if inCoreference:
            if not coreferenceStart:
                coreferenceStart = True
                for _ in sentences:
                    _ = nltk.word_tokenize(_)
                    newSentences.append(_)
                
                sentences = copy.deepcopy(newSentences)
    
            s = re.match('\s+\(([0-9]+),([0-9]+),\[([0-9]+),([0-9]+)\]\) -> \(([0-9]+),([0-9]+),\[([0-9]+),([0-9]+)\]\)', line)
            ssid    = int(s.group(1))-1
            swStart = int(s.group(3))-1
            swEnd   = int(s.group(4))-1
    
            esid    = int(s.group(5))-1
            ewStart = int(s.group(7))-1
            ewEnd   = int(s.group(8))-1
            
            
            newSentences[ssid][swStart:swEnd] = sentences[esid][ewStart:ewEnd]
    
    ## write output
    for sent in newSentences:
        fout.write(re.sub('\s+([.?!])$', "\g<1>", " ".join(sent)))
        fout.write(" ")

    # close files
    fin.close()
    fout.close()

            

## Read the stories, and generate new stories with coreference
## resolved.

fin = open(sys.argv[1], 'r')
for story in fin:
    story = story.strip()
    data = re.split('\t', story)
    name = data[0]
    print 'Reading story %s' %(name)
    storyText = formatForPrint(data[2])
    cwd = os.getcwd()
    os.chdir(corenlpDir)
    print(os.getcwd())
    tempInputFile = open("gen_corenlp_temp_input.txt", 'w')
    tempInputFile.write(storyText)
    tempInputFile.close()
    print 'Running ', cmd, " -file gen_corenlp_temp_input.txt -outputFormat text"
    os.system(cmd+" -file gen_corenlp_temp_input.txt -outputFormat text")
    generate_story('gen_corenlp_temp_input.txt.out', cwd+'/'+outDir+'/'+name+'.aftercoref.txt')
    os.chdir(cwd)

fin.close()

