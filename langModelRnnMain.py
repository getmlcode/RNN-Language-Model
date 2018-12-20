from sklearn.model_selection import train_test_split
import numpy as np
import preprocessText
import RNNTrain
import time
import pickle

saveDir = "D:\\Acads\\IISc ME\\Projects\\NeuralLanguageModel\\"
def languageModel(sentenceData,sentenceLabel,xTest,yTest,vocabSize,\
                  learning_rate=0.005,\
                  nepoch=50,\
                  evaluate_loss_after=5):
    
    print('Initializing weight matrices')
    U,V,W = RNNTrain.initWeightMatrices(vocabSize,100)
    #U = np.load(saveDir+"U.npy")
    #V = np.load(saveDir+"V.npy")
    #W = np.load(saveDir+"W.npy")
    losses = []
    num_examples_seen = 0
    print("\n\nTraining in progress")

    start = time.time()
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = RNNTrain.calculate_loss(sentenceData,sentenceLabel,\
                                           U,V,W)
            losses.append((num_examples_seen, loss))
            print('Number of examples = {}, Loss = {}, Epoch = {}'.\
                  format(num_examples_seen,loss,epoch))

            if ( len(losses)>2 and losses[-1][1] > losses[-2][1] \
                 and losses[-1][1] > losses[-3][1]):
                learning_rate = learning_rate * 0.5

        end = time.time()

        if (end-start) > 3600:
            print('Saving After 1 or more hour training')
            testLoss = RNNTrain.calculate_loss(xTest,yTest,U,V,W)
            np.save(saveDir+"U_time"+str(end)+str(testLoss)+'_'+str(num_examples_seen)+".npy",U)
            np.save(saveDir+"V_time"+str(end)+str(testLoss)+'_'+str(num_examples_seen)+".npy",V)
            np.save(saveDir+"W_time"+str(end)+str(testLoss)+'_'+str(num_examples_seen)+".npy",W)
            start = time.time()
        
        print('Epoch {} in progress '.format(epoch+1))
        for i in range(len(sentenceLabel)):
            dLdU,dLdV,dLdW = RNNTrain.Backprop_TT(sentenceData[i],\
                                                  sentenceLabel[i],\
                                                  U,V,W,4)
            U -= learning_rate*dLdU
            V -= learning_rate*dLdV
            W -= learning_rate*dLdW
            num_examples_seen += 1
                
    return U,V,W,losses[-1]

if __name__=="__main__":
    allSentences = \
    preprocessText.getSentencesFromFiles("C:\\DataSets\\NeuralLM")
    #print('\n-----\n'.join(sentences))
    print('Encoding sentences as array')
    X_train,Y_train,index_to_word,word_to_index = \
    preprocessText.sentenceToDataArray(allSentences,4500,\
                                       'START_SENT','END_SENT',\
                                       'ThisIsMissing')
    
    with open(saveDir+"index_to_word.txt", "wb") as fp:
        pickle.dump(index_to_word, fp)
        
    print("\n===Data Summary===")
    print("{} Sentences extracted ".format(len(allSentences)))

    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,\
                                                        test_size=.3)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    for i in range(5):
        U,V,W,lossOnTrain = languageModel(X_train,Y_train,X_test,Y_test,len(index_to_word))
        print('\nInit Point {} , Train Loss {}',i,lossOnTrain)
        print('U=\n{}\nV=\n{}\nW=\n{}'.format(U.shape,V.shape,W.shape))
        
        lossOnTest = RNNTrain.calculate_loss(X_test,Y_test,U,V,W,4)
        print('\nIter {} , Test Loss {}',i,lossOnTest)
        np.save(saveDir+"U_Complete"+str(lossOntest)+".npy",U)
        np.save(saveDir+"V_Complete"+str(lossOntest)+".npy",V)
        np.save(saveDir+"W_Complete"+str(lossOntest)+".npy",W)
