# Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search

We present a prompting framework called **LLMCS** that leverages large language models to perform few-shot conversational query rewriting for conversational search. 

We explore three prompting methods to generate multiple query rewrites and hypothetical responses, and propose to aggregate them into an integrated representation that can robustly represent the user’s real contextual search intent.

## Environment
We conduct experiemnts in Python 3.8.13.

Main packages:
- Generating rewrites:
    - openai==0.26.4
    - numpy==1.23.5

- For evaluation: 
    - torch==1.12.0+cu113
    - trec-car-tools==2.6
    - faiss-gpu
    - pyserini==0.17.0


```python
conda create -n llmcs python=3.8
source activate llmcs
pip install -r requirements.txt
```



## Data

We provide the preprocessed cast19 and cast20 datasets in the `datasets` folder.

`demonstrations.json` contains four exemplars randomly sampled from the CAsT-22 datasets. We manually write CoT for all of its turns.


## Running
LLMCS contains three prompting methods, including *Rewriting Prompt (REW)*, *Rewriting-Then-Response Prompt (RTR)*, and *Rewriting-And-Response Prompt (RAR)*. We also design chain-of-thought tailored to conversational searhc intent understanding that can be incorporated into these prompting methods.

First, you should set your OpenAI API key in `generator.py`
```python
# TODO: Write your OpenAI API here.
OPENAI_KEYS = [
    'Your key',
]
```

### REW Prompting 
To perform REW prompting, run:
```shell
bash scripts/run_prompt_rewrite.sh
```
Also, you can enable CoT by running:
```shell
bash scripts/run_prompt_cot_rewrite.sh
```

### RTR Prompting 
Similarly, to perform RTR prompting, run:
```shell
bash scripts/run_prompt_rewrite_then_response.sh
```
Note that you need to provide a pre-generated rewrite file (i.e., `rewrite_file_path`) for running `prompt_rewrite_then_response.py`. To enable CoT for RAR, you can set `rewrite_file_path` to the rewrite file generated using CoT.
```sh
--rewrite_file_path="./results/cast20/REW/rewrites.jsonl" \
# --rewrite_file_path="./results/cast20/COT-REW/rewrites.jsonl" \ +COT
```


### RAR Prompting 
```shell
bash scripts/run_prompt_rewrite_and_response.sh
```
Also, you can enable CoT by running:
```shell
bash scripts/run_prompt_cot_rewrite_and_response.sh
```


## Results
A `rewrites.jsonl` file, which contains the rewrites and hypothetical responses,  will be generated into the `work_dir` that you set in the running script.


We have provided our generated `rewrites.jsonl` files in the `results` folder.

The Keys of `rewrites.jsonl`:
- predicted_rewrite: rewrite (list)
- preidcted_response: hypothetical response (list)
- other auxiliary information


