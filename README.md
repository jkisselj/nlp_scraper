# NLP Scraper

Проект для сбора новостей с выбранного источника (у нас — Variety) и их обогащения с помощью NLP:

- извлечение организаций (spaCy NER, тип ORG),
- определение темы статьи (topic classifier, `results/topic_classifier.pkl`),
- анализ тональности (NLTK VADER),
- детекция “экологического скандала” через семантическую близость к списку ключевых слов,
- сохранение результата в единый CSV.

## Структура проекта

```text
.
├── data/
│   ├── articles_YYYY-MM-DD.jsonl   # результаты скрейпера
│   ├── topic_train.csv             # обучающие данные для классификатора тем
│   └── topic_test.csv              # тестовые данные для классификатора тем
├── results/
│   ├── training_model.py           # скрипт обучения классификатора тем
│   ├── topic_classifier.pkl        # обученная модель
│   ├── learning_curves.png         # (создаётся, если данных достаточно)
│   └── enhanced_news.csv           # обогащённые новости
├── scraper_news.py                 # скрейпер с variety.com
├── nlp_enriched_news.py            # NLP-движок
├── requirements.txt
└── README.md

 ## Установка окружения

 python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab

1. Сбор новостей
python scraper_news.py

Скрипт пройдётся по нескольким разделам Variety и сохранит статьи в data/articles_YYYY-MM-DD.jsonl с полями:

-id
-url
-date
-headline
-body

Сейчас скрипт обходит несколько страниц и может собрать >200 статей за запуск.

2. Обучение классификатора тем

python results/training_model.py

Скрипт:

-читает data/topic_train.csv и data/topic_test.csv;
-обучает TF-IDF + LogisticRegression;
-пытается построить learning curve (если хватит данных);
-сохраняет модель в results/topic_classifier.pkl.
-⚠️ Если данных мало, скрипт пропустит построение графика и просто сохранит модель.

3. NLP-обогащение новостей

python nlp_enriched_news.py

Скрипт:

-Загружает последние собранные статьи из data/.
-Извлекает ORG через spaCy.
-Подгружает модель тем из results/topic_classifier.pkl.
-Считает тональность через NLTK VADER.
-Считает “scandal_distance”: берёт предложения с организациями и сравнивает их с эмбеддингами экологических ключевых слов -(pollution, deforestation, ...), берёт максимум.
-Сортирует по Scandal_distance и помечает top-10 статей как Top_10=True.
-Сохраняет всё в results/enhanced_news.csv.

Структура CSV:

-uuid
-URL
-date
-headline
-body
-Org (список строк)
-Topics (список строк)
-Sentiment (float, VADER compound)
-Scandal_distance (float)
-Top_10 (bool)
-Объяснение выбора эмбеддингов и метрики
-Для детекции скандалов использована модель sentence-transformers/all-MiniLM-L6-v2, потому что:
-она достаточно лёгкая, чтобы запускать локально;
-даёт sentence-level эмбеддинги;
-поддерживается библиотекой sentence-transformers.

Для сравнения использовано cosine similarity (util.cos_sim), т.к. это стандартная метрика для сравнения нормализованных эмбеддингов. В качестве итоговой метрики по статье берётся максимум similarity между ключевыми словами и предложениями, содержащими упомянутые организации.

