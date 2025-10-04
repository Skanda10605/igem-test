DeepMethyGene Local Analysis Tool
This repository contains a command-line tool to run the DeepMethyGene prediction and analysis pipeline locally. It takes a target gene, pre-processed methylation data, and a pre-trained model to generate plots and data for several in-silico experiments.
Prerequisites
1. Python 3.8+: Ensure you have Python installed on your system. You can check by running python --version.
2. Git: You need Git to clone this repository.
Setup
1. Clone the Repository:
Open your terminal or command prompt and clone this repository to your local machine.
git clone <your-repository-url>
cd <your-repository-name>

2. Install Dependencies:
This project uses several Python libraries. They are listed in requirements.txt. Install them using pip:
pip install -r requirements.txt

3. Directory Structure:
For the script to work with its default settings, you must arrange your data and model files in the following structure within the cloned repository directory:
.
├── run_analysis.py         <-- The main script you will run
├── requirements.txt
├── README.md
|
├── data/                   <-- Create this directory for input data
│   ├── m_arrays_for_edit.csv
│   ├── mapped_filteredgenes_data.csv
│   └── hg19_promoter.txt
│
└── Gene Wise Model Weights/  <-- Place your model weights directory here
   ├── GENE1.pt
   ├── GENE1.json
   ├── GENE2.pt
   ├── GENE2.json
   └── ...

Usage
The main script is run_analysis.py. You run it from your terminal, providing the gene you want to analyze as an argument.
Basic Usage:
To run the analysis for the gene SLC7A5 using the default file paths shown above, use the following command:
python run_analysis.py --gene SLC7A5

The script will automatically:
   1. Load all necessary data files.
   2. Select the first available sample for the specified gene.
   3. Load the corresponding pre-trained model (SLC7A5.pt).
   4. Run three analysis tasks.
   5. Save the resulting plots (.png) and data (.csv) to the predictions_out/ directory.
Advanced Usage (Specifying Custom Paths):
If your files are located elsewhere, you can specify their paths using command-line arguments:
python run_analysis.py \
   --gene YOUR_GENE \
   --long-csv /path/to/your/m_arrays.csv \
   --mapped-csv /path/to/your/mapped_data.csv \
   --promoter-file /path/to/your/promoters.txt \
   --weights-dir /path/to/your/models/ \
   --output-dir /path/to/save/results/

Forcing CPU Usage:
If you have a GPU but want to force the script to use the CPU, add the --cpu flag:
python run_analysis.py --gene SLC7A5 --cpu

Output
The script will create an output directory (default: predictions_out/) containing:
   * {GENE}_task1_scenarios.png: A bar plot showing the 4-scenario prediction.
   * {GENE}_task2_window_sweep.png: A line plot for the hypomethylation window sweep.
   * {GENE}_task2_window_sweep.csv: The data corresponding to the window sweep plot.
   * {GENE}_task3_level_sweep.png: A line plot for the promoter methylation level sweep.
   * {GENE}_task3_level_sweep.csv: The data corresponding to the level sweep plot.