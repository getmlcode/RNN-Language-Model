import numpy as np

def initWeightMatrices(wordDim,hiddenDim=100):
    U = np.random.uniform(-np.sqrt(1./wordDim) \
                          ,np.sqrt(1./wordDim) \
                          ,(hiddenDim, wordDim))
    
    V = np.random.uniform(-np.sqrt(1./hiddenDim) \
                          ,np.sqrt(1./hiddenDim) \
                          ,(wordDim, hiddenDim))
    
    W = np.random.uniform(-np.sqrt(1./hiddenDim) \
                          ,np.sqrt(1./hiddenDim) \
                          ,(hiddenDim, hiddenDim))

    return U,V,W

def forward_prop(sentence,U,V,W):
    numOfWords = len(sentence)
    hidStates = np.zeros((numOfWords+1,W.shape[0]))
    out = np.zeros((numOfWords,V.shape[0]))

    for w in range(numOfWords):
        hidStates[w] = np.tanh(U[:,sentence[w]]+ \
                               W.dot(hidStates[w-1]))
        out[w] = softmax(V.dot(hidStates[w]))

    return [out,hidStates]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_loss(sentence,Y,U,V,W):
    TotWords = np.sum((len(y_i) for y_i in Y))
    Loss=0
    for i in np.arange(len(Y)):
        out,hidState = forward_prop(sentence[i],U,V,W)
        correct_word_predictions = out[np.arange(len(Y[i])), Y[i]]
        Loss += -1*np.sum(np.log(correct_word_predictions))
    return Loss/TotWords

def Backprop_TT(X,Y,U,V,W,bptt_truncate):
    out,hidStates = forward_prop(X,U,V,W)
    
    dLdU = np.zeros(U.shape)
    dLdV = np.zeros(V.shape)
    dLdW = np.zeros(W.shape)
    delta_o = out
    delta_o[np.arange(len(Y)), Y] -= 1.
    for t in np.arange(len(Y))[::-1]:
        dLdV += np.outer(delta_o[t], hidStates[t].T)
        delta_t = V.T.dot(delta_o[t])*(1-(hidStates[t]**2))
        for bptt_step in np.arange(max(0,t-bptt_truncate),t+1)[::-1]:
            dLdW += np.outer(delta_t, hidStates[bptt_step-1])
            dLdU[:,X[bptt_step]] += delta_t
            delta_t = W.T.dot(delta_t)*(1-hidStates[bptt_step-1]**2)
    return [dLdU, dLdV, dLdW]
    
    
    
  
if __name__=="__main__":
    '''Functionality Test Code'''
    U,V,W = initWeightMatrices(5,3)

    print('U=\n{}\nV=\n{}\nW=\n{}'.format(U,V,W))

    out,hidStates = forward_prop([0,2,3,1,4,2],U,V,W)

    print('out =\n{}\nStates =\n{}\nPredicted Sentence=\n{}'.\
          format(out,hidStates,np.argmax(out, axis=1)))

    dLdU,dLdV,dLdW = Backprop_TT([0,2,3,1,4,2],[2,3,1,4,2,3],U,V,W,3)

    print('dLdU =\n{}\ndLdV =\n{}\ndLdW =\n{}'.\
          format(dLdU,dLdV,dLdW))
    

    
    
