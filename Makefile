.PHONY: setup build-paper clinical imaging ensemble evaluate clean

setup:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt
	python3 config.py

build-paper:
	cd results && latexmk -pdf -interaction=nonstopmode -halt-on-error final.tex

clinical:
	python3 run_ddi_as.py --mode clinical --clinical_data data/clinical --output_dir results

imaging:
	python3 run_ddi_as.py --mode imaging --as_data data/mri/as --healthy_data data/mri/healthy --output_dir results

ensemble:
	python3 run_ddi_as.py --mode ensemble --output_dir results

evaluate:
	python3 run_ddi_as.py --mode evaluate --output_dir results

clean:
	cd results && latexmk -C final.tex || true