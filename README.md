# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

---

## TEAM MEMBERS
- Janez Tomšič
- Žan Pušenjak
- Matic Zadobovšek

---

## PROJECT DESCRIPTION

This project focuses on *automatic generation of Slovenian traffic news* using Natural Language Processing (NLP) techniques. The goal is to develop a system that transforms raw traffic data from promet.si into structured news reports that follow the predefined format used in radio broadcasting. This will reduce the manual workload for editors while also maintaining consistency and accuracy in traffic reports.

To achieve this, we explore *pre-trained language models*, *structured prompting*, and *fine-tuning*.

---

## PROJECT STRUCTURE
- **data/RTVSlo/** ... provided dataset we are working with
  - **Podatki - rtvslo.si** ... rtf files
    - **Promet 2022**
      - *Januar 2022*
      - ...
      - *December 2022*
    - **Promet 2023**
    - **Promet 2024**
  - *Podatki - PrometnoPorocilo_2022_2023_2024.xlsx* ... Excel file with data from promet.si website
  - *PROMET, osnove.docx* ... instructions (shorter)
  - *PROMET.docx* ... instructions (detailed)
- **notebooks/**
  - *exploration.ipynb* ... Jupyter notebook for easier exploration of given data
- **report/**
  - *project_submission_1.pdf* ... report of the *first submission*
- *environment.yml* ... Conda environment configuration

---

## INSTALLATION & SETUP

To have a reproducible environment, we use Conda. Install dependencies using the provided ```environment.yml```:

```
conda env create -f environment.yml
conda activate nlp-project
```

Once the environment is set up, you can explore and run code inside the **notebooks/** directory.


