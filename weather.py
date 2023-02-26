from vosk import Model, KaldiRecognizer  # оффлайн-распознавание от Vosk
from googlesearch import search  # поиск в Google
from pyowm import OWM  # использование OpenWeatherMap для получения данных о погоде
from termcolor import colored  # вывод цветных логов (для выделения распознанной речи)
from dotenv import load_dotenv  # загрузка информации из .env-файла
import speech_recognition  # распознавание пользовательской речи (Speech-To-Text)
import googletrans  # использование системы Google Translate
import pyttsx3  # синтез речи (Text-To-Speech)
import wikipediaapi  # поиск определений в Wikipedia
import random  # генератор случайных чисел
import webbrowser  # работа с использованием браузера по умолчанию (открывание вкладок с web-страницей)
import traceback  # вывод traceback без остановки работы программы при отлове исключений
import json  # работа с json-файлами и json-строками
import wave  # создание и чтение аудиофайлов формата wav
import os  # работа с файловой системой

owm = OWM('7bb42489685bf0f5121ada1582da14bf')
print(owm.supported_languages)
mgr = owm.weather_manager()
print(mgr)
observation = mgr.weather_at_place('Paris, FR')
print(observation)
observation.weather.detailed_status  # Nuageux
print(observation.weather.detailed_status)

open_weather_map = OWM('d81cbce2c8aa1dc195aa1f9b7324507a')

# запрос данных о текущем состоянии погоды
weather_manager = open_weather_map.weather_manager()
observation = weather_manager.weather_at_place('Orenburg')
weather = observation.weather
print(weather.detailed_status)
