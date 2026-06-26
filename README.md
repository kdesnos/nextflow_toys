# Nextflow Traces & Metrics Analyzer

This project provides tools to analyze, store, and model execution metrics (Time, RAM) of Nextflow pipelines from their HTML reports and log files (based on a Modified Nextflow). The ultimate goal is to model performance (via Amdahl's law and linear regressions) to inject these predictive metrics into prototyping tools such as PREESM.

---

# Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# UNIT-TEST
## with pytest (recommandé)
```bash
./tests/get_data.sh
pytest
```

---

# DOCUMENTATION 
## HTML documentation

```bash
pdoc src/ -o ./docs
```

---

### Project Structure

```text
nextflow_toys-main/
├── dat/                    
├── notebooks/              
├── tests/                  
├── src/                   
│   ├── __init__.py
│   ├── analysis/           
│   │   ├── amdahl_linear_regressor.py
│   │   ├── nf_trace_db_analyzer.py
│   │   └── ...
│   ├── db/                
│   │   ├── nf_trace_db_manager.py
│   │   ├── processes_table_manager.py
│   │   └── ...
│   ├── exporters/
|       ├── export_dag_to_pisdf.py
│   │   └── ...
│   └── extractors/
|       ├── extract_trace_from_html.py
│       └── ...
├── nextflow_trace_DB_creation_script.sql  
├── requirements.txt
└── README.md
```