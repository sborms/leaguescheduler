.PHONY: example

push:
	git add .
	git commit -m "$(m)"
	git push

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

web:
	uv run streamlit run app.py

example:
	rm -rf example/output
	uv run 2rr --input-file "example/input.xlsx" --output-folder "example/output" --seed 321 --n-iterations 5000

time:
	uv run timings.py

profile:
	python -m cProfile -o profile.out timings.py
	snakeviz profile.out
	
experiment:
	uv run 2rr \
		--input-file "experiments/Moeilijke reeksen.xlsx" \
		--output-folder "experiments/difficile" \
		--seed 505 \
		--n-iterations 10000
