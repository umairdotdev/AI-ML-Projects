
# Global Terrorism Analysis

This repository contains scripts and data for analyzing global terrorism trends.

## Contents
- **R Scripts**:
  - `GlobalTerrorismDatabase.R`: Analyzes global terrorism data for trends by year, region, and country.
  - `TerroristAttacksIreland.R`: Focuses on analyzing terrorism trends specific to Ireland.

- **Data**:
  - `globalterrorismdb.rar`: Compressed dataset containing global terrorism data from 1970 to 2016.

## Instructions
1. **Extract the Data File**:
   - Use a RAR extraction tool (e.g., WinRAR, 7-Zip, or any equivalent software) to extract the file `globalterrorismdb.rar`.
   - Place the extracted `globalterrorismdb.csv` file in the same directory as the scripts.

2. **Run the Scripts**:
   - Open R or RStudio and load the provided scripts.
   - Execute the scripts to analyze the data and generate visualizations.

3. **Review Outputs**:
   - Check the visualizations and summaries for insights into global terrorism trends.

## Prerequisites
- Ensure the following R libraries are installed before running the scripts:
  - `Matrix`, `ggplot2`, `data.table`, `treemap`, `highcharter`, `doMC`, `readxl`, `tidyr`, `dplyr`, `caret`.

## Outputs
- The scripts generate:
  - Visualizations and maps showing terrorism trends by region, country, and year.
  - Specific analyses for terrorism trends in Ireland.
