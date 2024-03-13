# llama_cdcr
This repository contains source code for reproducing the experiments that were conducted for the paper titled,
"Okay, Let's Do This! Modeling Event Coreference with Generated Rationales and Knowledge Distillation", accepted at NAACL 2024 Main Conference.
Our experiments use LLaMA2-chat-7B as the teacher model and Longformer-base as the student model. 

## Installation and Dependencies 
The [**requirements.txt**](./requirements.txt). file contains dependencies that are needed to run the experiments in our pipeline. 
In order to download the weights of the LLaMA2-chat-7B model (the teacher model), please use this
(https://ai.meta.com/resources/models-and-libraries/llama-downloads/) link from Meta. After downloading the weights, convert them to the 
HuggingFace Transformers format for accessing their pretrained model libraries using [**convert_llama_weights_to_hf.py**](./convert_llama_weights_to_hf.py)
For the Longformer-base model (student model), 
use the [HuggingFace link](https://huggingface.co/allenai/longformer-base-4096) to access the pretrained model.

## Reproduce main results in the paper
Please run this notebook [**Experiments_pipeline.ipynb**](./Experiments_pipeline.ipynb) to generate the 
full results table that includes all the models, datasets and ablations (Tables 1 and 7 in the paper). 

## Generating Inner Monologues for ECB+, GVC and AIDA Phase 1
The  [**generate_inner_monologues.py**](./generate_inner_monologues.py) generates the step-by-step FTRs (inner-monologues)
using LLaMA2-chat-7B. Use the function `create_zero_shot_prompts_for_eval(dataset = 'ecb')` 
to generate the FTRs for ECB+ ('gvc' for GVC and 'ldc' for AIDA Phase 1). In order to map each rationale to 
the corresponding event mentions in the corpora, use [**generate_inner_monologue_maps.py**](./generate_inner_monologue_maps.py.py). 
The generated inner-monologues for the experiments in this paper along with the mapping of gold label mentions can be found at this [Google Drive link](https://drive.google.com/drive/folders/1KiDIIDn5hxboqL3awVTHJVzHbrxSy-Th?usp=drive_link).

## Phase 1 Training: ROEC 
For training the Rationale-Oriented Event Clustering (ROEC) with the student model, run the script [**clustering_IM_rationale.py**](./clustering_IM_rationale.py). 
This training phase does not use the teacher model. 

## Phase 2 Training: Coreference Knowledge Distillation (CKD)
For training teh CKD phase, use the saved student model weights from the previous phase (ROEC)
and run the script [**training.py**](./training.py). 
## Evaluating the Student Model 
In order to predict pairwise scores with the trained student model and get final coreference metrics with the CoVaL scorer, 
use [**prediction.py**](./prediction.py). 

## Zero-shot Evaluation 
For LLaMA2-chat-7B zero-shot evauation, use the function `create_zero_shot_prompts_for_eval(dataset = 'ecb')` in  [**generate_inner_monologues.py**](./generate_inner_monologues.py).
For GPT 3.5-Turbo, use the notebook [**chatgpt.ipynb**](./chatgpt.ipynb)

## Clustering plots
Run [**cluster_analysis.py**](./cluster_analysis.py) to generate the distribution of clusters vs true positives for our Long (KD+ ROEC) model and the plain Longformer. 
## Human Evaluation of FTRs 
To generate the Krippendorff's alpha (α) and other metrics for our human evaluation component of the generated FTRs, run [**human_eval.ipynb**](./human_eval.ipynb)
 
 
