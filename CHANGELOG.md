## 1.0.0 (2026-05-29)

### Feat

- **submit-finetune-test.sh**: add a finetune testing script before a full finetune job is submitted
- **Makefile,-.env.example**: add ssh-key login to finetune-sync command
- **finetune/**: add finetune scripts and configs for running finetune job the Goethe HPC cluster

### Fix

- **gpt-oss-20b-lora.yaml**: gpt-oss -> gpt_oss
- **submit-finetune.sh,-submit-finetune-test.sh**: fix llama-factory cli overrides
- **pyproject.toml**: require torchaudio to pull from the pytorch-rocm source
- **cluster-setup.sh**: make uv install have --verbose output
- **cluster-setup.sh,-submit-setup.sh,-REAMDE.md**: build env on test cpu, update README accordingly
- **cluster-setpu.sh,-README.md**: update script and README to move uv cache to repo system so it can hardlink
- **cluster-setup.sh**: change huggingface-cli to hf
- **cluster-setup.sh,-README.md**: remove premature rocm module loading

### Refactor

- **src/**: restructure project

## 0.4.0 (2026-05-20)

### Feat

- **analyze_tokenizer_fertility.py**: analyze tokenizer fertility for coding use-case

## 0.3.0 (2026-05-18)

### Feat

- **upload_dataset.py**: update so both the base and instruct datasets are uploaded to hugginface
- **build_instruct_dataset.py**: generate the instruction dataset from the base dataset
- **instruction_generator.py**: add generate_instructions to randomly select from a list of predefined natural language instructions for converting one source language to one target language

### Refactor

- **build_base_dataset.py**: rename build_dataset.py to build_base_dataset.py

## 0.2.0 (2026-05-14)

### Feat

- **upload_dataset.py**: automate uploading dataset to huggingface
- **buld_dataset.py**: build dataset from leetcode solutions repo
