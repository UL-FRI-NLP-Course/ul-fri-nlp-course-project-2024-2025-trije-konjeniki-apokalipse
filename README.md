# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

---

## Team members  
- **Janez Tomšič**  
- **Žan Pušenjak**  
- **Matic Zadobovšek**  

---

## 📑 Table of contents
- [Project structure](#-project-structure)

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