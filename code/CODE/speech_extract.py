#Import libraries
import glob
import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


#Extract features
def extract_features(file_name):
    X, sample_rate = librosa.load(file_name)
    #Short time fourier transformation
    stft = np.abs(librosa.stft(X))
    #Mel Frequency Cepstra coeff (40 vectors)
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    #Chromogram or power spectrum (12 vectors)
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #mel scaled spectogram (128 vectors)
    mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    # Spectral contrast (7 vectors)
    contrast=np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    #tonal centroid features (6 vectors)
    tonnetz=np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

#parsing audio files
def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz= extract_features(fn)
            except Exception as e:
                print("Error encountered while parsing file: ", fn)
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split("\\")[8].split("-")[2])
    return np.array(features), np.array(labels, dtype = np.int)

#fn=glob.glob(os.path.join(main_dir, sub_dir[0], "*.wav"))[0]

#One-Hot Encoding the multi class labels
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels + 1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode=np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode

#Extracting features in X
#Storing labels in y
main_dir = r'C:\Users\YMTS0297\PycharmProjects\Emotion Recognition using speech\Datasets\Audio_Speech_Actors_01-24'
sub_dir = os.listdir(main_dir)
print("\nCollecting features and labels.")
print("\nThis will take some time.")
features, labels = parse_audio_files(main_dir, sub_dir)
print("\nCompleted")

#save features
np.save('X', features)
#one hot encode labels
labels = one_hot_encode(labels)
np.save('y', labels)

emotions=['neutral','calm','happy','sad','angry','fearful','disgused','surprised']
#labels2=np.array([emotions[labels[i]-1] for i in range(len(labels))])
#np.save('y2', labels2)

#Loading features and labels
#X=np.load('X.npy') #features
#y=np.load('y.npy') #labels

#Splitting the dataset
#train_X, test_X, train_y, test_y = train_test_split(X, y, test_size= 0.3,random_state=42)

#Parameters
#n_dim = train_X.shape[1]
#n_classes = train_y.shape[1]

#n_hidden_units_1 = n_dim
#n_hidden_units_2 = 400
#n_hidden_units_3 = 200
#n_hidden_units_4 = 100

#Define model
#def create_model(activation_function='relu', init_type='normal', optimizer='adam', dropout_rate=0.2):
#    model = Sequential()
#    #Layer 1
#    model.add(Dense(n_hidden_units_1, input_dim = n_dim, kernel_initializer=init_type, activation=activation_function))
#    #Layer 2
#    model.add(Dense(n_hidden_units_2,kernel_initializer=init_type, activation=activation_function))
#    model.add(Dropout(0.2))
#    #Layer 3
#    model.add(Dense(n_hidden_units_3, kernel_initializer=init_type, activation=activation_function))
#    model.add(Dropout(0.2))
#    #Layer 4
#    model.add(Dense(n_hidden_units_4, kernel_initializer=init_type, activation=activation_function))
#    model.add(Dropout(0.2))
#    #Output layer
#    model.add(Dense(n_classes, kernel_initializer=init_type, activation='softmax'))

    #Model compilation
#    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
 #   return model

#Create the model
#model=create_model()

# Model Training
#lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=20, min_lr=0.000001)
# Please change the model name accordingly.
#mcp_save = ModelCheckpoint('model/hello.h5', save_best_only=True, monitor='val_accuracy', mode='max')

#Train the model
#import time
#start=time.time()
#history = model.fit(train_X, train_y, epochs=200, batch_size=4, validation_data=(test_X, test_y), callbacks=[mcp_save, lr_reduce])
#end=time.time()
#print(end-start)

#Prediction
#predict=model.predict(test_X, batch_size=4)

#convert PREDICTED probabilities to emotions
#emotions=['neutral','calm','happy','sad','angry','fearful','disgused','surprised']
#y_pred=np.argmax(predict, axis=1)
#predicted emotions of test dataset
#predicted_emo=[]
#for i in range(test_y.shape[0]):
#    emo=emotions[y_pred[i]]
#    predicted_emo.append(emo)

#actual emotions of test dataset
#actual_emo=[]
#y_true=np.argmax(test_y, axis=1)
#for i in range(test_y.shape[0]):
#    emo=emotions[y_true[i]]
#    actual_emo.append(emo)

#Creating a confusion matrix
#cm=confusion_matrix(actual_emo, predicted_emo)
#index = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
#columns = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
#cm_df = pd.DataFrame(cm,index,columns)
#plt.figure(figsize=(10,10))
#sns.heatmap(cm_df, annot=True)

#Training accuracy
accuracy=accuracy_score(actual_emo, predicted_emo)

#Plots
# Plotting the Train Valid Loss Graph

#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy'+str(max(history.history['val_accuracy'])))
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()




