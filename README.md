Feedback model - Predicting Effective Arguments
==============================
Установка необходимых зависимостей и активация среды:

```commandline
conda create --name <env> --file requirements.txt
conda activate <env>
```

Подготовка данных:
В папке `data/raw` должны находится необходимые сырые данные

```dvc repro```

Создание необходимых docker images:

```commandline
docker build -f Docker/app_image/Dockerfile -t model_app .
docker build -f Docker/mlflow_image/Dockerfile -t mlflow_server .
```

docker compose

```commandline
docker compose up -d
```

Обучение модели классификатора:
```commandline
python src/mpdels/train_model.py data/processed/train_split.csv data/processed/test_split.csv
```
Во время обучения создаются чекпоинты в папке `models/`. Обученные модели логируются на `MLflow` сервер.
После обучения, при необходимости, переведите получившуюся модель в стадию Production в `Mlflow UI`

Загрузка модели в сервис проводится автоматически, при необходимости перезагрузите контейнер `app`

Сервис готов к использованию, документация располагается по адресу `http:<your-ip>/docs`

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `conda list --export > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to preprocess data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

