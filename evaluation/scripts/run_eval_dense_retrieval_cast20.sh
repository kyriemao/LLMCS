eval_field_name="predicted_rewrite"
work_dir="../results/mkl_6"

eval_file_path="$work_dir/rewrites.jsonl" \
index_path="/home/kelong_mao/cis_indexes/cast19-20/ance"
qrel_file_path="/home/kelong_mao/cis_datasets/cast20/preprocessed/cast20_qrel.tsv"
retrieval_output_path="$work_dir/ance/+q+r+sc"

export CUDA_VISIBLE_DEVICES=1
python eval_dense_retrieval.py \
--eval_file_path=$eval_file_path \
--eval_field_name=$eval_field_name \
--qrel_file_path=$qrel_file_path \
--index_path=$index_path \
--retriever_path="/home/kelong_mao/PLMs/ance-msmarco" \
--use_gpu_in_faiss \
--n_gpu_for_faiss=1 \
--top_n=1000 \
--rel_threshold=2 \
--retrieval_output_path=$retrieval_output_path \
--include_query \
--include_response \
--aggregation_method="sc" \
