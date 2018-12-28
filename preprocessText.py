import os
import nltk
import nltk.data as ntd
import itertools
import numpy as np

def getSentencesFromFiles(dataDir):
    '''Read text files in a directory
       and return a list of all sentences
       Doing this for huge data will be suicidal
    '''
    sentDetector = ntd.load('tokenizers/punkt/english.pickle')
    allSentences=[]
    for folderName, subfolders, filenames in os.walk(dataDir):
        for file in filenames:
            print("Extracting Sentences from file ",file)
            text = open(dataDir+'\\'+file,encoding='utf-8')
            sentence = sentDetector.tokenize(text.read())
            allSentences.extend(sentence)
    return allSentences

def sentenceToDataArray(sentenceList,vocabSize,startToken,endToken,unknownToken):
    
    sentences = ["%s %s %s"%(startToken, sent, endToken) \
                 for sent in sentenceList]
    tokenizedSentences = [nltk.word_tokenize(sent) \
                           for sent in sentences]
    wordFreq = nltk.FreqDist(itertools.chain(*tokenizedSentences))

    print("\n===Text Summary===")
    print ("Found {} unique words tokens ".format(len(wordFreq)))
    vocab = wordFreq.most_common(vocabSize-1)
    print ("The least frequent word {}".format(vocab[-1]))
    print ("The most frequent word {}".format(vocab[0]))
    
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknownToken)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    for i, sent in enumerate(tokenizedSentences):
        tokenizedSentences[i] = [w if w in word_to_index \
                                 else unknownToken for w in sent]
    
    X_train = np.asarray([[word_to_index[w] \
                           for w in sent[:-1]] \
                           for sent in tokenizedSentences])
    Y_train = np.asarray([[word_to_index[w] \
                           for w in sent[1:]] \
                           for sent in tokenizedSentences])

    return X_train,Y_train,index_to_word,word_to_index

if __name__=="__main__":
    allSentences = \
    getSentencesFromFiles("C:\\DataSets\\NeuralLM")
    
    print("===Sentences===")
    #print('\n-----\n'.join(sentences))
    print("Number Of Sentences : {}".format(len(allSentences)))

    X_train,Y_train,index_to_word,word_to_index = \
    sentenceToDataArray(allSentences,4500,'STAT_SENT','END_SENT','ThisIsMissing')
    print(X_train.shape,Y_train.shape)
