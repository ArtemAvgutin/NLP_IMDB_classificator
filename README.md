# NLP_IMDB_classificator
## Binary classification project. Determining positive and negative movie reviews.
## Задача бинарной классификации. Определение положительных и отрицательных отзывов на фильмы.
<p align="center">
   <img src="https://d26oc3sg82pgk3.cloudfront.net/files/media/edit/image/48110/article_full%402x.jpg" width="600" height="300">
</p>

## About (Eng)
* Target:
Create your own neural network and LSTM using vector representations and one hot encoding to determine the sentiment of movie reviews
* Result:
Neural networks were created with a sentiment determination accuracy of ~83% and ~85%, respectively.

## About (Rus)
* Цель:
Создать собственную нейронную сеть и LSTM с использованием векторных представлений и one hot encoding для определения тональности отзывов на фильмы
* Результат:
Созданы нейронные сети с точностью определения тональности ~83% и ~85% соответсвенно.

# Немного информации о проекте
Отзывы делятся на:
* положительные (оценка >= 7) 1 - отзыв положительные (25000 тыс.)
* отрицательные (оценка <= 4) 0 - отзыв отрицательные (25000 тыс.)

## Визуализация плотного векторного представления слов / Visualization of dense vector representation of words
![image](https://github.com/ArtemAvgutin/NLP-IMDB-classificator/assets/131138862/a1af2a24-38c8-4325-9b87-1fbb1afcaa4a)
## Обучение созданной нс / Training created nn
![image](https://github.com/ArtemAvgutin/NLP-IMDB-classificator/assets/131138862/677771b3-9005-477e-b558-3e643ff86571)
## Обучение сети LSTM / Training an LSTM network
![image](https://github.com/ArtemAvgutin/NLP-IMDB-classificator/assets/131138862/c5a67b78-9cc1-4d7e-93d2-938a0601bc7a)

Была использована токенизация текста, чтобы мы могли расшифровать любой отзыв и найти любое слово по индексу.
Также использовалось плотное векторное прелставление и его визуализация по которым можно определять эмоциональную окраску текста.
# Some info about project
Reviews are divided into:
* positive (score >= 7) 1 - positive review (25,000)
* negative (score <= 4) 0 - negative review (25,000)

Text tokenization was used so that we could decipher any review and find any word by index.
A dense vector representation and its visualization were also used, which can be used to determine the emotional coloring of the text.
