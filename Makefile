brush:
	ruff check --select I --fix .
	ruff format .

lint:
	ruff check .

install:
	pip install uv
	uv venv
	uv sync

freeze:
	uv sync
	uv pip compile pyproject.toml -o requirements.txt >/dev/null

example:
	cmd /c rmdir /s /q example_output
	2rr --input_file "example_input.xlsx" --output_folder "example_output" --seed 505 --n_iterations 100 --clip_upp 40

web:
	uv run streamlit run app.py

experiment:
	uv run 2rr --config_file "experiments/config.json"

time:
	uv run timings.py

profile:
	python -m cProfile -o profile.out timings.py
	snakeviz profile.out