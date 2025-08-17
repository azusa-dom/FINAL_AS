.PHONY: setup build-paper paper-release report-tables report-figs clinical imaging ensemble evaluate clean

setup:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt
	python3 config.py

build-paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error results/final.tex

paper-release:
	latexmk -pdf -interaction=nonstopmode -halt-on-error docs/paper/final_release.tex

report-tables:
	python3 src/reporting/build_report_assets.py --input data/processed --outdir results/reports

report-figs:
	python3 src/reporting/plot_figures.py --input data/processed --outdir results/figures_generated

clinical:
	python3 run_ddi_as.py --mode clinical --clinical_data data/clinical --output_dir results

imaging:
	python3 run_ddi_as.py --mode imaging --as_data data/mri/as --healthy_data data/mri/healthy --output_dir results

ensemble:
	python3 run_ddi_as.py --mode ensemble --output_dir results

evaluate:
	python3 run_ddi_as.py --mode evaluate --output_dir results

clean:
	latexmk -C results/final.tex || true
	latexmk -C docs/paper/final_release.tex || true