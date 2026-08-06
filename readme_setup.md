# Setup for reproducing experiments for the IRSE Project

First there is a general `setup.sh` script that creates the needed directories in part_1 and part_2. We need an inputs and results directory. In inputs we are going to have the datasets and the queries. In results the rag pipeline will write results file with evaluation and generated answers from the LM. These will be different for part 1 and 2.

The `.sh` scripts need to be given execution permission before running:

```bash
chmod +x setup.sh
```

This script will setup a python environment `.venv` if there isn't one that is already setup. If you already have a python environment with a different name, just change the `ENV_NAME` in the script to match your environment.

Then the requirements for running the project will be installed using the `requirements.txt` file.

It's important to note that this project should be ran in a machine with a GPU and cuda support.

**ATTENTION**: The `setup.sh` script should be ran from the project root folder.

Also before running the scripts for 1 and 2 make sure to have the virtual environment activated.

## Part 1

For part 1 the inputs folder should contain - `irse_documents_2026_recipes.parquet` and `irse_queries_2026_recipes.json`. If these files are not present in the folder they will be downloaded by the `run_part1.sh` script, otherwise the download will be skipped.

**ATTENTION**: This script runs the entire flow related to part 1 and should be run explicitly from the part_1 folder. 

Any configurations regarding the experiments can be changed from `config.py`, but by default all the values are the ones used for the baseline experiments.

Running the full pipeline for part 1 should produce 3 file:
- evaluation_FINAL.json - containing the evaluation metrics
- retrieved_docs_FINAL.json - containing the retieved documents for each query from the test set
- generated_answers_FINAL.json - containing the prompts and generated answers for each query

The amount of logs that the pipeline prints could be controlled from the `VERBOSE` variable in the config file.

I have left all the results file that I have generated before submission with the default configuration options.

## Part 2

For part 2 the inputs folder should contain - `acl_anthology_full.parquet`, `acl_anthology_queries.json` and `acl_anthology_queries.parquet`. If these files are not present in the folder they will be downloaded by the `run_part2.sh` script, otherwise the download will be skipped.

**ATTENTION**: This script runs the entire flow related to part 2 and should be run explicitly from the part_2 folder. 

Any configurations regarding the experiments can be changed from `config.py`, but by default all the values are the ones used for the baseline experiments. The setting to be most wary of is the `RUN_EVALUATION` variable, this controls if the full evaluation of the IR should run. By default this is is set to `False` and the evaluation will not run if not changed. The evaluation computes metric for every single combination of retrival strategy + query augmentation. In total those are 9 different retrieval pipelines:
- Dense (Regular) + no augmentation
- Dense (Regular) + query rewriting
- Dense (Regular) + HyDE
- Chunks + no augmentation
- Chunks + query rewriting
- Chunks + HyDE
- Hierarchical + no augmentation
- Hierarchical + query rewriting
- Hierarchical + HyDE

If you decide to run the full evaluation for part 2, beware it can take around an hour.

Running the full pipeline for part 2 with evaluation should produce 2 files in the `results` folder:
- evaluation_results_ALL.json
- generation_results_chunks_rewrite.json

The pipeline with evaluation will also create one folder with detailed retireval results called - `evaluation_details`. The additional folder can be ignored, that was used mainly to get more insights into the retrieval.

Here again, I have left all the results file that I have generated before submission with the default configuration options.

## Time estimates

For both part 1 and 2 the `main.py` is split into steps like "Step 1: Loading data", "Step 2: Preprocessing", etc. Under each sction there is a print statement that prints the time that the corresponding step takes and under it is an estimate of the time that the particular step will take.

Keep in mind this was a ran on a system with these specifications:

    CPU: Intel Core i9-14900HX @ 2.419GHz
    GPU: NVIDIA GeForce RTX 4070 Laptop GPU
    Memory: 32 GiB