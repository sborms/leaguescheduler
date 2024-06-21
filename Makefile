brush:
	isort .
	black .

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

freeze:
	pip freeze --local > requirements.txt

2rr:
	2rr --file "example_input.xlsx" --output_folder "example_output" --seed 505 --n_iterations 100

web:
	streamlit run app.py