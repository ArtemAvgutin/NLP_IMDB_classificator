# NLP-IMDB-classificator
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
Создать собственную нейронную сеть и LSTM с использованием векторных представлений и one hot encoding определять тональность отзывов на фильмы
* Результат:
Созданы нейронные сети с точностью определения тональности ~83% и ~85% соответсвенно.

# Немного информации о проекте / Some info about project
Отзывы делятся на:
* положительные (оценка >= 7) 1 - отзыв положительные (25000 тыс.)
* отрицательные (оценка <= 4) 0 - отзыв отрицательные (25000 тыс.)

Была использована токенизация текста, чтобы мы могли расшифровать любой отзыв и найти любое слово по индексу.
Также использовалось плотное векторное прелставление и его визуализация по которым можно определять эмоциональную окраску текста.
