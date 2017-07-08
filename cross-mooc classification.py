from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import cPickle,string
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.embeddings import Embedding 
from keras.layers.convolutional import Conv1D, MaxPooling1D 
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
# set parameters: 
maxlen = 500 
batch_size = 32 
filters = 250 
kernel_size = 5 
hidden_dims = 250 
epochs = 3
nb_epoch_t = 50

file=open("/home/a938/weixiaocong/mooc/cross-mooc/result.txt",'a')

def train_model(model,X_train_s,y_train_s,X_test_t,y_test_t):
    rmsprop = RMSprop(lr=0.0005, decay=1e-6, rho=0.9)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    #early_stopping=EarlyStopping(monitor='val_loss',patience=10,mode='min')
    model.load_weights('/home/a938/weixiaocong/mooc/cross-mooc/cnn_lstm-confusion/DS1-3')
    #model.fit(X_train_s,y_train_s, batch_size=batch_size,epochs=epochs,validation_data=(X_test_t,y_test_t),shuffle=True)
    #model.save_weights('/home/a938/weixiaocong/mooc/cross-mooc/cnn_lstm-confusion/DS1-3-google')
    #score = model.evaluate(Xt_train[519:],yt_train[519:], batch_size=batch_size, verbose=0) 
    #print('new dataset Test score:', score[0]) 
    #print('new dataset Test accuracy:', score[1])    
    #file.write("confusion:DS1 no TL to DS2:"+str(score[1])+"\n") 

    

def train_model_t(model,X_train_s,y_train_s,X_test_t,y_test_t):
    rmsprop = RMSprop(lr=0.0005, decay=1e-6, rho=0.9)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"]) 
    model.fit(X_train_s,y_train_s, batch_size=batch_size, 
           epochs=nb_epoch_t,validation_data=(X_test_t,y_test_t))
    score = model.evaluate(Xt_train[2592:],yt_train[2592:], batch_size=batch_size, verbose=0)    
    print('new dataset Test score:', score[0]) 
    print('new dataset Test accuracy:', score[1]) 
    file.write("confusion:DS1 TL to DS2:"+str(score[1])+"\n") 
    
print('Loading data...')

f=open('/home/a938/weixiaocong/mooc/cross-mooc/DS1-Confusion-selfvector.pkl','rb')
X, y = cPickle.load(f)
Xs_train=X[:]
ys_train=y[:]
f.close()

f=open('/home/a938/weixiaocong/mooc/cross-mooc/DS2-Confusion-selfvector.pkl','rb')
X, y = cPickle.load(f)
Xt_train=X[:]
yt_train=y[:]
f.close()


print('Pad sequences (samples x time)') 
Xs_train = sequence.pad_sequences(Xs_train, maxlen=maxlen)
Xt_train = sequence.pad_sequences(Xt_train, maxlen=maxlen)


wf=open('../dic-mooc-selfvector.p','rb')
revs, W, W2, word_idx_map, vocab=cPickle.load(wf)


print('Setting up Arrays for Keras Embedding Layer...')
n_symbols = len(word_idx_map) + 1  # adding 1 to account for 0th index
print ('n_symbols:',n_symbols)
print ('word_idx_map:',len(word_idx_map))
embedding_weights = np.zeros((n_symbols, 200))
for word,index in word_idx_map.items():
    embedding_weights[index, :] = W[index]
print ("embedding_weights:",len(embedding_weights)) 



feature_layers=[
    Embedding(output_dim=200,input_dim=n_symbols,weights=[embedding_weights],input_length=maxlen ),
    Dropout(0.25),     
    ]
classification_layers=[    
    Conv1D(filters=filters, 
                         kernel_size=kernel_size, 
                         padding='valid', 
                         activation='relu', 
                         strides=1),              
    LSTM(100),               
    Dropout(0.25),    
    Dense(1,kernel_regularizer=l2(3)),
    Activation('sigmoid') 
    ]    
print('Build model...') 
model = Sequential() 
  
for l in feature_layers+classification_layers:
    model.add(l)
train_model(model,Xs_train[:8888],ys_train[:8888],Xs_train[8888:],ys_train[8888:])

for l in feature_layers:
    l.trainable=False
#train_model_t(model,Xt_train[:494],yt_train[:494],Xt_train[494:988],yt_train[494:988])#DS1
#train_model_t(model,Xt_train[:260],yt_train[:260],Xt_train[260:519],yt_train[260:519])#DS2
#train_model_t(model,Xt_train[:151],yt_train[:151],Xt_train[151:303],yt_train[151:303])#DS3
#train_model_t(model,Xt_train[:2469],yt_train[:2469],Xt_train[2469:4938],yt_train[2469:4938])#DS1
train_model_t(model,Xt_train[:1296],yt_train[:1296],Xt_train[1296:2592],yt_train[1296:2592])#DS2
#train_model_t(model,Xt_train[:758],yt_train[:758],Xt_train[758:1515],yt_train[758:1515])#DS3
file.close()





