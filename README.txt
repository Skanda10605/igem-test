iGEM DeepMethyGene Analysis Tool
This project provides a command-line tool to analyze the relationship between DNA methylation and gene expression, based on the models and methods from the iGEM project.

The tool allows users to specify one or more genes, and then automates the following workflow:

Data Generation: Prepares the necessary methylation and expression datasets for the specified genes.

Model Training: Trains a custom Convolutional Neural Network (CNN) for each specified gene to predict its expression from methylation data.

Prediction & Analysis: Uses the trained models to perform several analyses, including:

Predicting gene expression under baseline and manipulated methylation conditions.

Generating plots to visualize the effect of sweeping methylation levels across different genomic window sizes.

Creating plots to show the impact of varying promoter methylation levels on gene expression.

Prerequisites
Before you begin, ensure you have Python 3 installed on your system. You will also need to install the required Python libraries.

1. Install Python Libraries
You can install all the necessary libraries using pip and the provided requirements.txt file. Open your terminal or command prompt and run:

pip install -r requirements.txt

2. Prepare Your Data Files
You must have the following data files in the same directory as the main.py script:

hg19_promoter.txt

BRCA_gene_exp_integrated.csv

BRCA_meth_integrated_filtered.csv

BRCA_data_meth_range.csv

integrated_gene_names_with_expression_new.csv

Usage
To run the software, navigate to the project directory in your terminal and execute the main.py script:

python main.py

The script will then prompt you to enter the names of the genes you wish to analyze.

Example
Run the script:

python main.py

You will be prompted for input. Enter your desired genes, separated by commas:

Enter the gene(s) you want to train and analyze (comma-separated): SLC7A5, TYMS, PTEN

The script will then proceed with the automated workflow: data generation, training, and prediction. Progress will be displayed in the terminal.

Output
The script will create two new directories in your project folder:

Gene Wise Model Weights/: This directory will contain the trained model files for each gene (.pt files) and their corresponding metadata (.json files).

predictions_out/: This directory will contain the results of the analysis, including:

CSV Files: Detailed prediction results for each analysis task.

PNG Images: Plots visualizing the results of the methylation sweeping analyses.

You can then inspect these files to see the results of your analysis.
