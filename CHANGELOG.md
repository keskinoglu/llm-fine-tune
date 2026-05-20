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
