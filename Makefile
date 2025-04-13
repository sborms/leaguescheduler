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
	2rr --input-file "example_input.xlsx" --output-folder "example_output" --seed 505 --n-iterations 100 --clip-upp 40

web:
	uv run streamlit run app.py

experiment:
	uv run 2rr \
		--input-file "experiments/input.xlsx" \
		--output-folder "experiments/output" \
		--seed 505 \
		--n-iterations 1000 \
		--m 7 \
		--p 1000 \
		--r-max 5 \
		--clip-bot 0 \
		--clip-upp 50

time:
	uv run timings/timings.py

profile:
	python -m cProfile -o profile.out timings/timings.py
	snakeviz profile.out