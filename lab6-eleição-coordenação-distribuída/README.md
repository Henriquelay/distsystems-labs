# How to run

Have the queue server running:

```shell
docker compose up -d
```

Use poetry: https://python-poetry.org/docs/#installation and install the requirements

```shell
poetry install
```


After that, should be as simple as running the terminals with the amount of nodes you desire. Default is set to 10, but you can define it in the last line of [main.py](/src/main.py). You can run inside poetry env with `poetry run python src/main.py`
