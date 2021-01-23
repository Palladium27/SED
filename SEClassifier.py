import time
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adamax
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import matplotlib.pyplot as plt
features=pd.read_csv('C:\\Users\\raval\\soundevent\\features\\onevall.csv',delimiter=',',header=None)
#testfeatures=pd.read_csv('C:\\Users\\raval\\project\\features\\level2\\testnewW1sym1.csv',delimiter=',',header=None)
#valfeatures=pd.read_csv('C:\\Users\\raval\\project\\features\\level2\\justfault2rbio1.1.csv',delimiter=',',header=None)
#valY=features[42][:]
#valoutput=pd.get_dummies(valY)
#Xval=valfeatures.iloc[:,0:42]

Y=features[14][:]
output=pd.get_dummies(Y)

#Ytest=testfeatures[42][:]
#yt=pd.get_dummies(Ytest)
#print(yt)
#Xtest=testfeatures.iloc[:,0:36]
X=features.iloc[:,0:14]
X_train, X_test, Y_train, Y_test = train_test_split(X,output, test_size=0.2)
model8 = Sequential([
    Dense(100, input_shape=(14,),kernel_initializer='lecun_normal'),
    Activation('tanh'),
    #Dense(128,kernel_initializer='lecun_normal'),
    #Activation('relu'),
    Dense(50,kernel_initializer='lecun_normal'),
    Activation('tanh'),
    Dense(18,kernel_initializer='lecun_normal'),
    Activation('tanh'),
    Dense(2,kernel_initializer='lecun_normal'),
    Activation('softmax'),
    
])

ada = Adamax(lr=0.01,decay=1e-6)
model8.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])
#model8.fit(X_train,Y_train,batch_size=len(X),validation_data=(X_test,Y_test),epochs=100)
mdl=model8.fit(X_train,Y_train,batch_size=len(X_train),validation_data=(X_test,Y_test),epochs=1000)
#model8.fit(X,output,batch_size=len(X),validation_data=None,epochs=200)
A=model8.predict(X_train, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
B=np.asarray(A)
ll=[(np.argmax(B[i])+1) for i in range(len(B))]
#print(ll)
#history.
gh=mdl.history['acc']
hg=mdl.history['val_acc']
plt.plot(mdl.history['acc'])
plt.plot(mdl.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#print(gh)
#print(hg)