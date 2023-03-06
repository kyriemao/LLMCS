from IPython import embed
import os
import json
import time
import argparse
from tqdm import tqdm, trange
from promptor import RewriteThenResponsePromptor
from generator import OpenAIGenerator, OPENAI_KEYS
from utils import set_seed, get_finished_sample_ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demonstration_file_path", type=str, required=True)
    parser.add_argument("--rewrite_file_path", type=str, required=True, help="rewrite file path")
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True, help='output rewrite path.')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_completion", type=int, default=1, help='the number of completions for generation. If n > 1, we only consider the first rewrite.')
    parser.add_argument("--open_ai_key_id", type=int, choices=[0,1,2,3,4,5], required=True)
    parser.add_argument("--seed", type=int, default=7)
    
    
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "parameters.txt"), "w") as f:
        params = vars(args)
        f.write(json.dumps(params, indent=4))
        
    return args


def main():
    args = get_args()    
    set_seed(args) 
    
    # model and promptor setting
    promptor = RewriteThenResponsePromptor()
    
    # only one completion for one rewrite. To be extended to N competions for one rewrite
    model_kwargs = {"n": args.n_completion, "top_p": 1, "temperature": 0.7, "max_tokens": 512, "logprobs": 1, "stop": promptor.stop_tokens}
    openai_key = OPENAI_KEYS[args.open_ai_key_id]
    generator = OpenAIGenerator(args.model_name, openai_key, **model_kwargs)
    
    # demos
    with open(args.demonstration_file_path, "r") as f:
        demo_dialogs = json.load(f)
    demo_prompt = promptor.build_demo_prompt(demo_dialogs)
   
    # existing predicted rewrites for each turn
    rewrite_dict = {}
    with open(args.rewrite_file_path, "r") as f:
        data = f.readlines()
    for line in data:
        line = json.loads(line)
        rewrite_dict[line['sample_id']] = line['predicted_rewrite']
    
    
    # test_dataset    
    output_file_path = os.path.join(args.work_dir, "rewrites.jsonl")
    finished_samples = get_finished_sample_ids(output_file_path)
    with open(args.test_file_path, "r") as f:
        test_dialogs = json.load(f)
    begin_time = time.time()
    
    # predict
    with open(output_file_path, "a+") as f:
        for i in trange(len(test_dialogs)):
            dialog = test_dialogs[i]
            conv_id = dialog['conv_id'] 
            turns = dialog['turns']
            
            pre_prompt = demo_prompt
            this_turn_predicted_rewrite, last_response = None, None
            for j in trange(len(turns)):
                turn_id = turns[j]['turn_id']
                sample_id = "{}_{}".format(conv_id, turn_id)
                if sample_id in finished_samples:
                    last_response = turns[j]['response']
                    if last_response == "":
                        last_response = response_list = finished_samples[sample_id]['predicted_response'][0]
                    continue

                # prompting
                this_turn_predicted_rewrite_list = rewrite_dict["{}_{}".format(conv_id, turn_id)]
                if args.n_completion > 1:   # we do not consider generate n * n_completion responses now.
                    this_turn_predicted_rewrite_list = [this_turn_predicted_rewrite_list[0]]
                all_response_list = []
                for this_turn_predicted_rewrite in tqdm(this_turn_predicted_rewrite_list):
                    prompt = promptor.build_this_turn_prompt_for_prediction(pre_prompt, turns[j], this_turn_predicted_rewrite, last_response)                    
                    while True:
                        try:
                            response_list = generator.generate(prompt, promptor.parse_returned_text)
                            all_response_list.extend(response_list)
                        except ValueError as e:
                            print("{}, re-generating...".format(e.args[0]))
                            continue
                        break

                last_response = turns[j]['response']
                if last_response == "":
                    last_response = response_list[0]
                pre_prompt = prompt
                
                # recording
                record = {}
                record['sample_id'] = sample_id
                record['predicted_rewrite'] = this_turn_predicted_rewrite_list
                record['predicted_response'] = all_response_list
                
                if sample_id not in finished_samples:
                    f.write(json.dumps(record))
                    f.write('\n')

                # output info
                print("{}, sample_id: {} \n exisiting_predicted rewrite: {} \n predicted response: {}".format(args.work_dir, sample_id, this_turn_predicted_rewrite_list[0], all_response_list[0]))

    print("{} Generation ok!, time cost {}".format(args.work_dir, time.time() - begin_time))

if __name__ == '__main__':
    main()
