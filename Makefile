PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) scripts/make_demo_dataset.py --rows 2500 --output data/demo_flows.csv
	$(PYTHON) scripts/benchmark_models.py --dataset data/demo_flows.csv --max-rows 2500 --reduced-k 10 --output results/demo_benchmark.csv

test:
	$(PYTHON) -m pytest -q

benchmark:
	$(PYTHON) scripts/benchmark_models.py --dataset data/CICIoT2023.csv --output results/ciciot2023_benchmark.csv

run-dashboard:
	streamlit run app/dashboard.py
