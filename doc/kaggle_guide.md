# Руководство: LLM Classification Finetuning

## Соревнование

**Цель:** предсказать предпочтения пользователей в попарном сравнении ответов LLM (A wins / B wins / Tie).

**Источник данных:** ChatBot Arena (человеческие предпочтения).

**Метрика:** log loss (чем меньше, тем лучше).

**Текущий топ:** ~0.83039

---

## Шаг 1: Регистрация и настройка

### 1.1 Создать аккаунт Kaggle

1. Зайти на [kaggle.com](https://www.kaggle.com)
2. Зарегистрироваться (или войти)

### 1.2 Установить Kaggle API

```bash
uv pip install kaggle
```

### 1.3 Получить API-токен

1. Перейти на [kaggle.com/account](https://www.kaggle.com/account) → вкладка "API"
2. Нажать "Create New Token"
3. Скачать файл `kaggle.json`
4. Положить в:
   - `~/.config/kaggle/kaggle.json` (Linux/Mac)
   - `C:\Users\<user>\.kaggle\kaggle.json` (Windows)

```bash
# Проверить
kaggle competitions list
```

---

## Шаг 2: Скачать данные

```bash
# Создать папку
mkdir -p data

# Скачать данные соревнования
kaggle competitions download -c llm-classification-finetuning -p data/

# Распаковать
unzip -d data/ llm-classification-finetuning.zip
```

**Структура данных:**

| Файл | Описание |
|------|----------|
| `train.csv` | Обучающая выборка с метками |
| `test.csv` | Тестовая выборка (нужно предсказать) |
| `sample_submission.csv` | Пример файла submission |

**Колонки train.csv:**
- `id` — уникальный идентификатор
- `model_a`, `model_b` — названия моделей
- `prompt` — вопрос к моделям
- `response_a`, `response_b` — ответы моделей
- `winner_model_a` — 1 если A победила, иначе 0
- `winner_model_b` — 1 если B победила, иначе 0
- `winner_tie` — 1 если ничья, иначе 0

**Колонки test.csv:**
- `id`, `prompt`, `response_a`, `response_b`

---

## Шаг 3: Подготовить окружение

```bash
# Создать виртуальное окружение
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Установить зависимости
uv pip install -r requirements.txt
```

---

## Шаг 4: Тренировать модель

### Вариант A: Запустить весь пайплайн (TF-IDF → SBERT → DeBERTa → Ensemble)

```bash
python train.py
```

Выход: `output/submission.csv`

### Вариант B: Поэтапно через ноутбуки

```bash
# Ноутбуки в папке notebooks/
# 01_eda.ipynb         — анализ данных
# 02_tfidf_baseline.ipynb  — TF-IDF baseline
# 03_sbert_baseline.ipynb  — SBERT baseline
# 04_deberta_finetuning.ipynb — DeBERTa finetuning
# 05_ensemble_submission.ipynb — ensemble и submission
```

---

## Шаг 5: Создать submission

### Формат файла

`submission.csv` должен содержать вероятности для каждого класса:

| Колонка | Описание |
|---------|----------|
| `id` | ID из test.csv |
| `winner_model_a` | вероятность что A победила |
| `winner_model_b` | вероятность что B победила |
| `winner_tie` | вероятность ничьей |

**Важно:** сумма вероятностей в строке должна = 1.0

```python
# Пример проверки в Python
import pandas as pd
sub = pd.read_csv('output/submission.csv')
row_sums = sub[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
print(f"Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}")  # должны быть ~1.0
```

---

## Шаг 6: Загрузить submission

```bash
# Проверить лидерборд
kaggle competitions leaderboard llm-classification-finetuning -p

# Отправить submission
kaggle competitions submit llm-classification-finetuning \
    -f output/submission.csv \
    -m "TF-IDF + SBERT + DeBERTa ensemble"

# Проверить статус
kaggle competitions submissions llm-classification-finetuning
```

---

## Подходы и ожидаемые результаты

| Approach | Описание | Target log loss |
|----------|----------|-----------------|
| TF-IDF + LogReg | Быстрый baseline | ~1.0 |
| SBERT + LogReg | Sentence-BERT embeddings | ~0.95 |
| DeBERTa-v3-base | Finetuning с position bias mitigation | ~0.87 |
| Ensemble | Комбинация всех моделей | < 0.84 |

---

## Ключевые техники для высокого результата

1. **Position bias mitigation** — случайный обмен response_a/response_b во время обучения (50% вероятность)
2. **Label smoothing** — предотвращает переобучение на 100% уверенность
3. **Mixed precision (fp16)** — ускоряет обучение, экономит память
4. **Cosine LR scheduler + warmup** — стабильная сходимость
5. **Temperature scaling** — калибровка предсказаний

---

## Возможные проблемы

| Проблема | Решение |
|----------|---------|
| "You must authenticate" | Проверить `kaggle.json` на месте |
| Data not found | Скачать заново: `kaggle competitions download -c llm-classification-finetuning` |
| CUDA out of memory | Уменьшить batch_size, использовать fp16 |
| Low score | Проверить submission format (вероятности, не классы) |

---

## Полезные ссылки

- [Kaggle Competition](https://www.kaggle.com/competitions/llm-classification-finetuning)
- [Top Solution Notebook](https://www.kaggle.com/code/ryqn00/llm-classification-finetuning)
- [GitHub Solution](https://github.com/carlacotas/llm-classification-finetuning)
- [Medium Writeup](https://medium.com/@carlacotas/my-first-kaggle-competition-llm-classification-finetuning-476db368b389)

---

## Чеклист перед отправкой

- [ ] Данные скачаны в `data/`
- [ ] `source .venv/bin/activate` выполнен
- [ ] `python train.py` завершился без ошибок
- [ ] `output/submission.csv` существует
- [ ] Проверено: сумма вероятностей в каждой строке = 1.0
- [ ] Kaggle API токен настроен
- [ ] Submission отправлен через `kaggle competitions submit`

---

*Обновлено: 2026-04-25*