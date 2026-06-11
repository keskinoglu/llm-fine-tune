## 1.2.0 (2026-06-11)

### Feat

- **evluation/**: add evaluation dataset along with evaluation-related infrastructure like the dockerfile
- **evluation/hpc/goethe**: add goethe hpc evaluation scripts
- **evaluation/**: add evaluation package, including a new evaluation dataset under the dataset/ phase
- **dataset/**: combine with secondary dataset from newfacade/LeetCodeDataset to add input/output pairs, problem difficulty, and a few other useful columns

### Fix

- **build_evaluation_dataset.py**: fix parsing error on input_output_pairs
- **goethe/env.sh**: place hugging face cache in node storage to avoid race condition on the locked file
- **configs/**: replace soon-to-be depricated warmup_ration: 0.3 -> warmup_steps: 10
- **goethe/env.sh**: export hugging face cache to work directory on setup

### Refactor

- **dataset/**: change terminology from problem -> code_snippet, i.e. reframe from leetcode problem to generic code_snippet
- **codebase**: refactor for easier maintainability and readabilitiy

## 1.1.0 (2026-05-31)

### Feat

- **submit-rebuild.env.sh**: add new shell script to submit an environment rebuilding job on the goethe cluster
- **submit-merge-and-publish.sh**: make card an optional parameter for the submit merge and publish call
- **publish/model_card/**: make model cards dynamically injectable during publishing
- **configs/**: add gemma3, mistral, and qwen2.5-coder configs
- **qwen-3.5-0.8b**: add configs for qwen 3.5 0.8b using the no_think template
- **publish/**: add lora adapter merging + huggingface publishing module as phase 4

### Fix

- **sumbit-rebuild-env.sh**: stop attempting to query GPUs on CPU-only node
- **submit-rebuild.env.sh,-submit-setup.sh**: bump SLRUM time limit from 15m to 1h
- **gemma-3**: change gemma-3-1b to gemma-3-4b to enable multimodal inpuu (the Llama Factory gemma3 template requires this)
- **goethe/**: update merge and publish commands
- **sumbit-merge.sh**: add version arg to submit-merge
- **goethe/**: moved the rocm load module to be in GPU-only training nodes
- **finetune/**: refactor to be cluster-agnostic, make llama-3.2 the new default target

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
