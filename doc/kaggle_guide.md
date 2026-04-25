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

## Шаг 3.5: Kaggle Notebooks (запуск в облаке)

Если нет GPU или хочешь работать в облаке:

### Вариант A: Kaggle Notebook

1. Открыть [kaggle.com/competitions/llm-classification-finetuning/code](https://www.kaggle.com/competitions/llm-classification-finetuning/code)
2. Нажать **New Notebook**
3. Скопировать код из `notebooks/` или `train.py` в ячейки
4. Подключить GPU: **Settings** → **Accelerator** → GPU (P100 / T4)
5. Запустить ячейки

### Вариант B: Kaggle API (клонирование ноутбука)

```bash
# Скачать готовый ноутбук с Kaggle
kaggle kernels pull <kernel-slug> -p ./notebooks/

# Или склонировать из GitHub
git clone https://github.com/<repo>/llm-classification-ft.git
cd llm-classification-ft
```

### Вариант C: Google Colab

```bash
# Загрузить файлы в Google Drive
# Или клонировать репозиторий в Colab:
!git clone https://github.com/<username>/llm-classification-ft.git
%cd llm-classification-ft
!pip install -r requirements.txt
```

В Colab: **Runtime** → **Change runtime type** → GPU T4 → **Save**

### Ноутбуки для Kaggle

| Ноутбук | Описание |
|---------|----------|
| `01_eda.ipynb` | Анализ данных, bias investigation |
| `02_tfidf_baseline.ipynb` | TF-IDF baseline |
| `03_sbert_baseline.ipynb` | SBERT baseline |
| `03b_tabpfn_sbert.ipynb` | TabPFN на SBERT embeddings |
| `04_deberta_finetuning.ipynb` | DeBERTa finetuning |
| `kaggle_submission.ipynb` | **Single notebook — все в одном** (для Kaggle submission) |

### Важно: Kaggle Notebooks лимиты

- GPU: 30 часов/неделю (P100) или 12 часов/неделю (T4)
- RAM: ~16 GB
- Нет доступа к файлам вне ноутбука — загружать данные через **Add Data**
- Интернет ограничен при приватных соревнованиях

---

## Kaggle Code Competition — Как отправить

Это **Code Competition** — все работает через Kaggle Notebooks. Нужно сделать **один notebook** со всем кодом.

### Подготовка

1. Скачать `kaggle_submission.ipynb` из папки `notebooks/`
2. Открыть [kaggle.com/competitions/llm-classification-finetuning/code](https://www.kaggle.com/competitions/llm-classification-finetuning/code)
3. Нажать **New Notebook** → **File** → **Upload Notebook** → загрузить `kaggle_submission.ipynb`

### Пошаговая инструкция

#### Шаг 1: Добавить данные

1. В ноутбуке нажать **Add Data** (справа)
2. Перейти на вкладку **Competition Data**
3. Найти `llm-classification-finetuning` → нажать **Add**
4. Данные появятся в `/kaggle/input/llm-classification-finetuning/`

#### Шаг 2: Подключить GPU

1. Нажать **Settings** (шестеренка справа)
2. **Accelerator**: выбрать **GPU T4** (или P100)
3. **Internet**: убедиться что включен (для скачивания models)
4. **Save**

#### Шаг 3: Проверить код

Код уже настроен на `/kaggle/input/` — ничего менять не нужно.

Ячейки выполняются по порядку:
1. Setup & Imports
2. Load Data
3. Phase 1: TF-IDF
4. Phase 2: SBERT (займет ~5 мин)
5. TabPFN (если данных < 10K)
6. Phase 3: DeBERTa (займет ~15 мин на T4)
7. Phase 4: Ensemble
8. Generate Submission

#### Шаг 4: Запустить все ячейки

1. Нажать **Run All** (Shift+Enter для одной ячейки)
2. Следить за прогрессом в tqdm bars
3. Время выполнения: ~30-40 минут на T4

#### Шаг 5: Commit и Submit

1. После успешного выполнения всех ячеек → нажать **Commit** (кнопка вверху справа)
2. Kaggle запустит ноутбук заново (без интернета, как при реальной проверке)
3. Дождаться результата commit (~10-15 минут)
4. После успешного commit → нажать **Submit**

### Проверка перед commit

Убедиться что:
- ✅ Ноутбук выполняется от начала до конца без ошибок
- ✅ В конце есть файл `submission.csv` в `/kaggle/working/`
- ✅ `submission.csv` содержит колонки: `id`, `winner_model_a`, `winner_model_b`, `winner_tie`
- ✅ Сумма вероятностей в каждой строке ≈ 1.0

### Возможные проблемы

| Проблема | Решение |
|----------|---------|
| Commit timeout (>9 часов) | Уменьшить DeBERTa epochs до 2, или batch_size до 4 |
| GPU OOM | Уменьшить MAX_LENGTH до 384, batch_size до 4 |
| TabPFN fails | Notebook уже обрабатывает — просто пропустит |
| "Internet disabled" на commit | Убедиться что модели уже загружены в session (run before commit) |

### Важно

- **Commit** — это реальная проверка (без инета, time-limited)
- **Submit** — отправляет результат commit в лидерборд
- Между commit и score может быть до 15 минут variance
- Можно делать несколько commits для улучшения

---

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
# 03b_tabpfn_sbert.ipynb   — TabPFN on SBERT embeddings
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