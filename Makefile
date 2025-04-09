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
	uv lock
	uv pip freeze > requirements.txt

2rr:
	cmd /c rmdir /s /q example_output
	2rr --input_file "example_input.xlsx" --output_folder "example_output" --seed 505 --n_iterations 100 --clip_upp 40

web:
	streamlit run app.py

experiment:
	2rr --config_file "experiments/config.json"