brush:
	ruff check --select I --fix .
	ruff format .
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
	uv run 2rr --input-file "example_input.xlsx" --output-folder "example_output" --seed 321 --n-iterations 500

web:
	uv run streamlit run app.py

experiment:
	uv run 2rr \
		--input-file "experiments/Moeilijke reeksen.xlsx" \
		--output-folder "experiments/difficile" \
		--seed 505 \
		--n-iterations 10000

time:
	uv run timings/timings.py

profile:
	python -m cProfile -o profile.out timings/timings.py
	snakeviz profile.out