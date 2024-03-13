# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, LSTM
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
# %matplotlib inline

max_words=10000 #Используем самые популярные 10000 слов, можно больше

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

x_train[11] #Токенизация слов

y_train[11] #Ответ рецензии 1 - положительный, 0 - отрицательный

word_index = imdb.get_word_index()

word_index

# Преобразуем словарь, чтобы по номеру получать слово
reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key

# Печатаем 35 самых частых слов
for i in range(1, 36):
    print(i, '---->', reverse_word_index[i])

index = 222
message = ''
for code in x_train[index]:
    word = reverse_word_index.get(code - 3, '!') # ! - Это замена символа для тех слов, которые не вошли в наш лимит, в моём случае, которые не вошли в 10000 выборки в самом начале
    message += word + ' '
message

maxlen = 200 #Максимальная длинна слов в отзыве, если слов менее 200, мы подставляем нули. Так-как нам нужна одинковая длинна, мы подгоняем все отзывы к 200 словам

# В функции обрезаем, если более 200 и добавляем символ заполнитель (0), если менее 200
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

x_train[1]

y_train[1]

model = Sequential()
model.add(Embedding(max_words, 2, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=15,
                    batch_size=128,
                    validation_split=0.1)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=1)

embedding_matrix = model.layers[0].get_weights()[0]

embedding_matrix[:5]

word_index_org = imdb.get_word_index()

word_index = dict()
for word,number in word_index_org.items():
    word_index[word] = number + 3
word_index["<Заполнитель>"] = 0
word_index["<Начало последовательности>"] = 1
word_index["<Неизвестное слово>"] = 2
word_index["<Не используется>"] = 3

word = 'good'
word_number = word_index[word]
print('Номер слова', word_number)
print('Вектор для слова', embedding_matrix[word_number])

reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key

filename = 'imdb_vectors_embeddings.csv'

with open(filename, 'w') as f:
    for word_num in range(max_words):
      word = reverse_word_index[word_num]
      vec = embedding_matrix[word_num]
      f.write(word + ",")
      f.write(','.join([str(x) for x in vec]) + "\n")

!head -n 20 $filename

files.download('imdb_vectors_embeddings.csv')

plt.scatter(embedding_matrix[:,0], embedding_matrix[:,1])

review = ['brilliant', 'fantastic', 'amazing', 'good',
          'bad', 'awful','crap', 'terrible', 'trash', 'worst']
enc_review = []
for word in review:
    enc_review.append(word_index[word])
enc_review

review_vectors = embedding_matrix[enc_review]
review_vectors

plt.scatter(review_vectors[:,0], review_vectors[:,1])
for i, txt in enumerate(review):
    plt.annotate(txt, (review_vectors[i,0], review_vectors[i,1]))

x_train[3]

def vectorize_sequences(sequences, dimension=10000): #Длинна вектора 10000
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. #Там, где у нас будет слово, ставим 1, а где нет ничего - вектор заполняется нулями
    return results

x_train = vectorize_sequences(x_train, max_words)
x_test = vectorize_sequences(x_test, max_words)

x_train[0][:50]

len(x_train[0])

y_train[0]

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(max_words,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=1)

print("Доля верных ответов на тестовых данных:", round(scores[1] * 100), '%')

max_words=10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
x_train[5002]

model = Sequential()
model.add(Embedding(max_words, 8, input_length=maxlen))
model.add(LSTM(32, recurrent_dropout = 0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=15,
                    batch_size=128,
                    validation_split=0.1)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=1)

print("Доля верных ответов на тестовых данных:", round(scores[1] * 100), '%')
