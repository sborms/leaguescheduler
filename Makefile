brush:
	isort .
	black .

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

freeze:
	pip freeze --local > requirements.txt

2rr:
	2rr --input_file "example_input.xlsx" --output_folder "example_output" --seed 505 --n_iterations 100

web:
	streamlit run app.py

experiment:
	2rr --config_file "experiments/config.json"