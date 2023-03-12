import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM , Embedding
from keras.models import Sequential
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from lib import cleansing

file_dataset = 'datasets/ready_datasets.csv'
df = pd.read_csv(file_dataset)

df = df[['content','label']]
df = df.dropna(subset=['content'])
df = df.dropna(subset=['label'])

df['content'] = df['content'].apply(cleansing)

X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], \
                                                    test_size=0.1, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])


train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
print("Training data size : ", train_df.shape)
print("Test data size : ", test_df.shape)

top_words = 20000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(train_df['content'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['content'])

max_review_length = 200
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
y_train = train_df['label']

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
# define early stopping to stop training after 5 epochs of not improving
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)

model.fit(X_train,y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[tensorboard, early_stopping])

list_tokenized_test = tokenizer.texts_to_sequences(test_df['content'])
X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)
y_test = test_df['label']
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)

model.save('output_model/lstm_model.h5')


