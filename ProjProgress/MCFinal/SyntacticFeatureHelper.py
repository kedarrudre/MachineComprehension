from collections import defaultdict
import nltk
from nltk.parse.stanford import StanfordDependencyParser

# Set up stanford NLP parser using NLTK. Please make sure you have those jar files are right place.
path_to_model = 'C:\StanfordCoreNLP\stanford-parser-3.7.0-models.jar'
path_to_jar = 'C:\StanfordCoreNLP\stanford-parser.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_model)

wh_words_set = set(['what', 'who', 'why', 'when', 'how', 'where', 'which'])
g_count = 0

class SyntacticFeatureHelper(object):
    def __init__(self):
        self.synt_cache = defaultdict()
        self.synt_cache ['dep_graph'] = {}
        self.synt_cache ['root_info'] = {}

    def getWhWord(self, q):
        for q_tkn in q:
            if q_tkn.lower() in wh_words_set:
                return q_tkn.lower()
        return None

    # NOTE - TODO: Ignores case. No option today to override.
    def question2stmt(self, raw_question, raw_a):
        raw_question = raw_question.lower()
        raw_a = raw_a.lower()
        raw_tokens = nltk.word_tokenize(raw_question)
        wh_word = self.getWhWord(raw_tokens)
        if wh_word is None:
            return None # No wh_word present in question.
        dep_graph = self.getDependencyGraph(raw_question)
        root_node = self.getRootNode(dep_graph, raw_question)    
        root_word, root_add, root_pos = root_node['word'], root_node['address'], root_node['tag']
    
        root_deps = {}
        for (k,v) in dep_graph.nodes[root_add]['deps'].iteritems():
            root_deps[k] = dep_graph.nodes[v[0]]['word'].lower()

    
        is_modified = False
        if '?' in raw_tokens:
            raw_tokens.remove('?')
    
        # Rule based question to statement conversion. 
        if wh_word == "what":
            if root_pos in ['VB', 'VBD', 'VBP', 'VBZ']:
                if 'dobj' in root_deps and root_deps['dobj'] == wh_word:
                    if 'nsubj' in root_deps and root_deps['nsubj']:
                        word = root_deps['nsubj']
                        raw_tokens.insert(raw_tokens.index(word) + 1, raw_a) # insert a after w. 
                        raw_tokens.remove(raw_tokens[raw_tokens.index(wh_word) + 1]) # remove word after c.
                        raw_tokens.remove(wh_word) # remove c.
                        is_modified = True
                if 'nsubj' in root_deps and root_deps['nsubj'] == wh_word and wh_word in raw_tokens:
                    raw_tokens[raw_tokens.index(wh_word)] = raw_a
                    is_modified = True
            elif root_pos == 'WP': # Replace c with a
                raw_tokens[raw_tokens.index(wh_word)] = raw_a
                is_modified = True
            elif root_pos == 'NN': 
                if 'nsubj' in root_deps and root_deps['nsubj'] == wh_word:
                    raw_tokens[raw_tokens.index(wh_word)] = raw_a
                    is_modified = True
        elif wh_word == "which": # Delete the word after c. Replace c with a
            raw_tokens.remove(raw_tokens[raw_tokens.index(wh_word) + 1]) # index overflow? #TODO
            raw_tokens[raw_tokens.index(wh_word)] = raw_a
            is_modified = True
        elif wh_word == "where":
            if root_pos in ['VB', 'VBP']:
                if 'advmod' in root_deps and root_deps['advmod'] == wh_word:
                    raw_tokens.remove(raw_tokens[raw_tokens.index(wh_word) + 1]) # delete the word after c.
                    raw_tokens.remove(wh_word) # delete c
                    if 'dobj' in root_deps and root_deps['dobj']:
                        word = root_deps['dobj']
                        raw_tokens.insert(raw_tokens.index(word) + 1, raw_a) # insert a after w.
                        is_modified = True
                    else: 
                        raw_tokens.insert(raw_tokens.index(root_word) + 1, raw_a) # insert a after r if w is not found.
                        raw_tokens.remove(root_word) # based on orignal Hai Wang's paper
                        is_modified = True
            elif root_word.lower() == 'is' and root_pos in ['VBZ'] and 'advmod' in root_deps and root_deps['advmod'] == wh_word \
            and 'nsubj' in root_deps and root_deps['nsubj']: # based on orignal Hai Wang's paper
                word = root_deps['nsubj']
                raw_tokens.remove(root_word) # move r after w. move => remove + insert.
                raw_tokens.insert(raw_tokens.index(word) + 1, root_word)
                raw_tokens.insert(raw_tokens.index(root_word) + 1, raw_a)
                is_modified = True
        elif wh_word == "who":
            if root_pos in ['NN', 'VB', 'VBD', 'VBG'] and 'nsubj' in root_deps and root_deps['nsubj'] == wh_word: # based on both the papers.
                raw_tokens[raw_tokens.index(wh_word)] = raw_a  # Replace c with a
                is_modified = True
            elif root_pos == 'WP':
                raw_tokens[raw_tokens.index(wh_word)] = raw_a  # Replace c with a
                is_modified = True
        # miscellaneous section:
        special_phrase1 = ["how many", "how much"]
        for phrase in special_phrase1:
            if phrase in raw_question.lower():
                return raw_question.replace(phrase, raw_a, 1).replace('?', '')
        if "when" in raw_question.lower():
            return raw_question.replace("when", "").replace('?', '') + " " + raw_a
        # miscellaneous section: ENDS
    
        if wh_word == "how":
            if root_pos == "VB" and 'nsubj' in root_deps and root_deps['nsubj']:
                word = root_deps['nsubj']
                raw_tokens = raw_tokens[raw_tokens.index(word):len(raw_tokens)]
                raw_tokens.insert(raw_tokens.index(word) + 1, raw_a) # insert a after w
                raw_tokens.insert(raw_tokens.index(word) + 1, 'to') # insert 'to' between w and a
                is_modified = True

        elif wh_word == "why":
            raw_tokens.remove(raw_tokens[raw_tokens.index(wh_word) + 1]) # delete the word after c.
            raw_tokens.remove(wh_word) # delete c
            raw_tokens.insert(len(raw_tokens), "because")
            raw_tokens.insert(len(raw_tokens), raw_a)
            is_modified = True

        if is_modified == True: # stmt generated.
            return " ".join(raw_tokens)
        return None # if unable to convert question into statement.

    def getSyntacticFeaturesScore(self, raw_question, raw_a, sent):
        global g_count
        print "called ", g_count
        g_count += 1
        # Ignore cases.
        raw_a = raw_a.lower()
        raw_question = raw_question.lower()
        sent = sent.lower()
        #if '.' in sent:
        #    sent.remove('.')

        stmt = self.question2stmt(raw_question, raw_a)
        if stmt is None: # question cannot be converted to statement using given rules.
            return (1,1,1,1)
        dep_graph_stmt = self.getDependencyGraph(stmt)
        dep_graph_sent = self.getDependencyGraph(sent)
    
        dep_count = 1 # counts matching dependencies/relations between statement and sentence
        root_ans_count = 1 # root and answer matches in dependency triplet
        only_root_count = 1 # only root matches in dependency triplet
        no_match_count = 1 # nothing matches in dependency triplet


        stmt_triples_verbose = dep_graph_stmt.triples()
        sent_triples_verbose = dep_graph_sent.triples()
    
        stmt_triples = []
        sent_triples = []
        for (w1_v, rel, w2_v) in stmt_triples_verbose:
            stmt_triples.append((w1_v[0], rel, w2_v[0]))
        for (w1_v, rel, w2_v) in sent_triples_verbose:
            sent_triples.append((w1_v[0], rel, w2_v[0]))

        root_stmt = self.getRootNode(dep_graph_stmt, stmt)
        root_word_stmt, root_add_stmt, root_pos_stmt = root_stmt['word'], root_stmt['address'], root_stmt['tag']
        #root_sent = self.getRootNode(dep_graph_sent, sent)

        for (u1, rel, v1) in stmt_triples:
            for (u2, rel, v2) in sent_triples:
                if (u1, rel, v1) == (u2, rel, v2):
                    dep_count += 1
            if u1 == raw_a and v1 == root_word_stmt:
                root_ans_count += 1
            elif v1 == root_word_stmt:
                only_root_count += 1
            else:
                no_match_count += 1

        return (dep_count, root_ans_count, only_root_count, no_match_count)

    def getRootNode(self, dep_graph, str):
        if str in self.synt_cache['root_info']:
            return self.synt_cache['root_info'][str] # we don't need to do deepcopy as we use it only for read.
        for node in dep_graph.nodes.values():
            if node['rel'] == 'root':
                 self.synt_cache['root_info'][str] = node
                 return node
        raise Exception('root node is None!')

    def getDependencyGraph(self, str):
        if str in self.synt_cache['dep_graph']:
            return self.synt_cache['dep_graph'][str] # we don't need to do deepcopy as we use it only for read.
        result  = dependency_parser.raw_parse(str)
        dep_graph = list(result)[0]
        self.synt_cache['dep_graph'][str] = dep_graph
        return dep_graph

## Test code. Remove later
#def test():
#    with open('raw_questsions.txt', 'r') as f:
#        for line in f:
#            if 'story:' in line:
#                print "\n" + line
#                continue
#            qa_pair = line.split('||')
#            ques = qa_pair[0].lower()
#            ans = qa_pair[1]
#            for a in ans.split(';'):
#                a = a.lower().strip()
#                sent = question2stmt(ques, a)
#                print ques + "__" + a, sent

#def test2():
#    ques = "what was the hardest thing for tom and his friends to fix?"
#    ans = ['door', 'house', 'window', 'toilet']
#    for a in ans:
#        stmt = question2stmt(ques, a)
#        print a + " || " + stmt
#        print getSyntacticFeaturesScore(ques, a, "Fixing the door was also easy but fixing the window was very hard")
#        print ""


#ques = "what color was the door that delta wanted to go in?"
#a = "red"
#sent = question2stmt(ques, a)
#print sent
#test()
#test2()
#print "End..."

