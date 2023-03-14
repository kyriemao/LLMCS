# 🎓 Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search

We present a prompting framework called **LLMCS** that leverages large language models to perform few-shot conversational query rewriting for conversational search. 

We explore three prompting methods to generate multiple query rewrites and hypothetical responses, and propose to aggregate them into an integrated representation that can robustly represent the user’s real contextual search intent.

## 🌏 Environment
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



## 📑 Data

1. We provide the preprocessed cast19 and cast20 datasets in the `datasets` folder.

2. `demonstrations.json` contains four exemplars randomly sampled from the CAsT-22 dataset. We manually write CoT for all of its turns.


## 🏃 Running
LLMCS contains three prompting methods, including *Rewriting Prompt (REW)*, *Rewriting-Then-Response Prompt (RTR)*, and *Rewriting-And-Response Prompt (RAR)*. We also design chain-of-thought tailored to conversational search intent understanding that can be incorporated into these prompting methods.

To get started, first, you should set your OpenAI API key in `generator.py`
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
Note that you need to provide a pre-generated rewrite file into the field of `rewrite_file_path` for running `prompt_rewrite_then_response.py`. To enable CoT for RAR, you can set `rewrite_file_path` to the rewrite file generated using CoT.
```sh
--rewrite_file_path="./results/cast20/REW/rewrites.jsonl" \
# --rewrite_file_path="./results/cast20/COT-REW/rewrites.jsonl" \ +COT
```


### RAR Prompting
Similarly, to perform RAR prompting, run: 
```shell
bash scripts/run_prompt_rewrite_and_response.sh
```
Also, you can enable CoT by running:
```shell
bash scripts/run_prompt_cot_rewrite_and_response.sh
```


## 🥚 Results
A `rewrites.jsonl` file, which contains the rewrites or/and hypothetical responses,  will be generated into the `work_dir` that you set in the running script.


We have provided our generated `rewrites.jsonl` files in the `results` folder.

The Keys of `rewrites.jsonl`:
- `predicted_rewrite`: rewrite (list)
- `preidcted_response`: hypothetical response (list)
- other auxiliary information



## ⚖️ Evaluation
We design three aggregation methods, including *MaxProb*, *Mean*, and *SC*, to get the final search intent vector. Then we perform dense retrieval with [ANCE (click to download)](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip) for evaluation.


**To perform evaluation, you should first build the dense index that contains all candidate passage embeddings.** There have been many good public repositories that instruct how to build this index, such as [AutoRewriter](https://github.com/thunlp/ConversationQueryRewriter), [ConvDR](https://github.com/thunlp/ConvDR), and [ConvTrans](https://github.com/kyriemao/ConvTrans). One can refer them to build the index. All passage embeddings generated from ANCE are around 103GB. 


Then, run the following script for evaluation:
```sh
cd evaluation
bash run_eval_dense_retrieval.sh
```
We have annotated the important arguments of `run_eval_dense_retrieval.sh` as below:
```sh
# eval_dense_retrieval.sh
# An example of evaluating RTR on cast20.
eval_field_name="predicted_rewrite"
work_dir="../results/cast20/RTR"    # set your the folder containing your `rewrites.jsonl`file

eval_file_path="$work_dir/rewrites.jsonl" \
index_path="" # set the pre-built index which contains all candidate passage emebddings. 
qrel_file_path="../datasets/cast20_qrel.tsv" # set the qrel file path
retrieval_output_path="$work_dir/ance/+q+r+sc" # set your expected output folder path

export CUDA_VISIBLE_DEVICES=0
python eval_dense_retrieval.py \
--eval_file_path=$eval_file_path \
--eval_field_name=$eval_field_name \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="" \ # set the pre-trained ANCE model path.
--use_gpu_in_faiss \
--n_gpu_for_faiss=1 \
--top_n=1000 \
--rel_threshold=2 \ # set 1 for cast19 and 2 for cast20.
--retrieval_output_path=$retrieval_output_path \
--include_query \
--include_response \ # enable `include_response` if you test RTR or RAR prompting.
--aggregation_method="sc" \ # you can set [`maxprob, mean, sc`]
```

Evaluation results that contains the following three files will be output into `retrieval_output_path` that you set.
- `metrics.res`: all evaluation metrics.
- `parameters.txt`: parameters record of the evaluation.
- `res.trec`: detailed TREC-style search results. 


### Main Results:
| Retriever |        System       |  CAsT-19  |  CAsT-20  |
|:---------:|:-------------------:|:---------:|:---------:|
|           |                     |   NDCG@3  |   NDCG@3  |
|    BM25   |       QuReTeC       |   0.338   |   0.171   |
|    BM25   |         T5QR        |   0.258   |   0.141   |
|    BM25   |        Manual       |   0.309   |   0.241   |
|    ANCE   |       QuReTeC       |   0.443   |   0.287   |
|    ANCE   |         T5QR        |   0.417   |   0.299   |
|    ANCE   |        Manual       |   0.461   |   0.422   |
|    ANCE   |        ConvDR       |   0.463   |   0.340   |
|    ANCE   |        COTED        |   0.478   |   0.342   |
|    ANCE   |         CRDR        |   0.472   |   0.350   |
|    ANCE   | **LLMCS  (RAR + Mean)** | **0.506** | **0.465** |
|           | RI over Manual        | 9.8\%     | 10.2\%    |
|           | RI over 2nd-Best      | 5.9\%     | 32.9\%    |


Table 1: "RI over Manual or 2nd-Best" means "relative improvements over Manual or the second-best method except for Manual".

## ✍️ Reference
If you use LLMCS in a research paper, please cite our work as follows:

@article{khattab2022demonstrate,
  title={Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search},
  author={Mao, Kelong and Dou, Zhicheng and Chen, Haonan and Mo, Fengran and Qian, Hongjin},
  journal={arXiv preprint arXiv:2303.06573},
  year={2023}
}

