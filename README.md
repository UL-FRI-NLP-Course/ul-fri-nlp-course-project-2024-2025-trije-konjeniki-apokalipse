# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

---

## Team members  
- **Janez Tomšič**  
- **Žan Pušenjak**  
- **Matic Zadobovšek**  

---

## 📑 Table of contents
- [Project structure](#-project-structure)
- [Reproducibility](#-reproducibility)
  - [Local setup with Conda](#local-setup-with-conda)
  - [📓 Notebooks](#-notebooks)
  - [🌐 Streamlit app](#-streamlit-app)
  - [💻 ARNES HPC cluster setup](#-arnes-hpc-cluster-setup)
---


## 📂 Project structure

Below is an overview of the repository layout.

```
.
├── report/
├── environment.yml
├── src/
│   ├── utils/
│   ├── notebooks/
│   │   ├── dataset_preparation.ipynb
│   │   ├── exploration.ipynb
│   │   ├── evaluation.ipynb
│   │   ├── finetuning_example.ipynb
│   │   └── prompting.ipynb
│   ├── evaluation/
│   │   ├── app/
│   │   │   └── app_evaluation.py
│   │   ├── progress/
│   │   └── results/
│   └── arnes_hpc/
│       ├── archive/
│       ├── containers/
│       ├── models/
│       │   └── final_model_9b/
│       └── scripts/
│           ├── finetuning.py
│           ├── instructions.txt
│           ├── run_base_model.py
│           ├── run_finetuned_model.py
│           ├── run_instructed_finetuned_model.py
│           ├── run_rag_model.py
│           ├── run_slurm_finetuning.sh
│           ├── run_slurm_base_eval.sh
│           ├── run_slurm_finetuned_eval.sh
│           ├── run_slurm_instructed_eval.sh
│           └── run_slurm_rag_eval.sh
│           └── rag/
│               ├── data_creation/
│               ├── embed.py
│               ├── rag_instructions_embeddings.npy
│               ├── rag_instructions.jsonl
│               ├── rag_roads_embeddings.npy
│               ├── rag_roads.jsonl
│               └── retrieve_example.py
└── README.md
```

---

### 📁 `report/`  
PDFs of our project write-ups and submission reports.

---

### 📄 `environment.yml`  
Conda environment for local reproducibility (`conda env create -f environment.yml`).

---

### 📂 `src/utils/`  
Helper functions for parsing, cleaning and structuring raw traffic data.

---

### 📂 `src/notebooks/`  
Interactive Jupyter notebooks for:  
- **dataset_preparation.ipynb** — building and cleaning our dataset 
- **exploration.ipynb** — data inspection and exploratory data analysis  
- **evaluation.ipynb** — automatic metrics (SloBERTa score) 
- **finetuning_example.ipynb** — a toy LoRA run to run on Colab  
- **prompting.ipynb** — structured-prompt experiments  

---

### 📂 `src/evaluation/`  
#### `app/`  
A Streamlit app (`app_evaluation.py`) we built to perform manual rating on test outputs.  
#### `progress/`  
Each member’s intermediate ratings.  
#### `results/`  
Final aggregated outputs for all **4 scenarios** (base-instructed, fine-tuned, fine-tuned + instructed, fine-tuned + instructed + RAG) on 500 examples.

---

### 📂 `src/arnes_hpc/`  
Everything needed to run our **full experiments on the ARNES HPC cluster**:

- **archive/**: old / discarded models & scripts  
- **containers/**: Singularity definition files for building `.sif` images  
- **models/final_model_9b/**: the GaMS-9B checkpoint & LoRA adapter  
- **scripts/**:  
  - **finetuning.py**: QLoRA fine-tuning on H100  
  - **instructions.txt**: base-prompt rules  
  - **run_*.py** & **run_slurm_*.sh**: launch scripts for each of the 4 experiments  
  - **rag/**: embed & index road-chunks & instruction docs with LaBSE + retrieval demo

---

## 🔁 Reproducibility

Follow these instructions depending on the setup (local or HPC):

---

### Local setup with Conda

You can start by creating the required environment using Conda. The provided `environment.yml` file will install all necessary dependencies under the environment name `nlp-project`. Later on we also provide separate instructions, for those who would prefer to manually install the needed dependencies.

```bash
conda env create -f environment.yml
conda activate nlp-project
```

---

### 📓 Notebooks

All notebooks under `notebooks/` are designed for local execution — with the exception of `finetuning_example.ipynb`, which is a simplified demo version of LoRA fine-tuning, and is prepared for execution on Colab.

To run notebooks locally, make sure the following Python packages are installed (already included via `environment.yml`, but if you would prefer to do manual installation):

```bash
pip install beautifulsoup4 matplotlib numpy pandas seaborn striprtf ipython transformers torch scikit-learn
```

---

### 🌐 Streamlit app

To run the manual evaluation Streamlit app in `src/evaluation/app/app_evaluation.py`, you'll need:

```bash
pip install pandas streamlit
```

You can then launch the appllication using:
```bash
cd src/evaluation/app
streamlit run app_evaluation.py
```

---

### 💻 ARNES HPC cluster setup

To run the full training or evaluation jobs on the ARNES HPC cluster:

1. Navigate to our project folder:
```bash
cd /d/hpc/projects/onj_fri/trije_konjeniki_apokalipse/
```

2. Launch jobs using:
```bash
sbatch run_slurm_<type>_eval.sh
```

The structure is modular — for each `.sh` (Slurm script) there's a corresponding Python script that performs the actual execution:

| SLURM script                   | Python script                      |
|-------------------------------|------------------------------------|
| `run_slurm_base_eval.sh`      | `run_base_model.py`               |
| `run_slurm_finetuned_eval.sh` | `run_finetuned_model.py`          |
| `run_slurm_instructed_eval.sh`| `run_instructed_finetuned_model.py`|
| `run_slurm_rag_eval.sh`       | `run_rag_model.py`                |

Each of these jobs produces a `.txt` file containing model outputs for 500 examples. These results are stored and used later for manual and automatic evaluation.

All the files required for running are already available under the same directory on HPC, so the workflow is fully reproducible and requires no extra setup.
