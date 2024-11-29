import argparse
import os

from d_trans import (tasks,
                    local_translate,
                    deepl_translate, 
                    utils,
                    )

cache_dir = os.environ['HF_HOME'] + '/hub'

def main():
    tasks_to_translate = list(tasks.task_registry.keys())
    langs = ["BG","DA","DE","ET","FI","FR","EL","IT","LV","LT","NL","PL","PT-PT","RO","SV","SK","SL","ES","CS","HU"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=None, help="comma separated list of tasks to translate. Can either be leaderboard tasks or links to jsonl files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deployment", type=str, default='vllm_local', choices=['vllm_local', 'hf_local', 'deepl'], help="Choose the deployment method")
    parser.add_argument("--model", type=str, default='haoranxu/ALMA-13B')
    parser.add_argument("--split", default="no_splits")
    parser.add_argument("--out_dir")
    parser.add_argument("--dummy", action="store_true", default=False)
    parser.add_argument("--lang", choices=utils.MultiChoice(langs), default=",".join(langs))
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--num_devices", type=int, default=1)
    args = parser.parse_args()
    print(args)

    if 'local' in args.deployment:
        if args.deployment == 'vllm_local':
            from llm.vllm_api import VLLM
            request_api = VLLM(model_name=args.model, num_devices=args.num_devices, max_model_len=4096, cache_dir=cache_dir)
        elif args.deployment == 'hf_local':
            from llm.huggingface_api import HuggingFaceLLM
            request_api = HuggingFaceLLM(model_name=args.model, cache_dir=cache_dir)
        else:
            raise ValueError("Please choose a valid deployment method.")

        local_translate.translate_tasks(tasks_to_translate=args.tasks,
                                        split=args.split,
                                        limit=args.limit,
                                        sample=args.sample,
                                        langs=args.lang,
                                        out_dir=args.out_dir,
                                        request_api=request_api,
                                        )
    elif args.deployment == 'deepl':
        cost = deepl_translate.translate_tasks(tasks_to_translate=args.tasks,
                                                split=args.split,
                                                limit=args.limit,
                                                sample=args.sample,
                                                langs=args.lang,
                                                dummy=args.dummy,
                                                out_dir=args.out_dir,
                                                compute_costs=True)
        resp = input(f"Translating under this configuration will incur a cost of {cost:.2f}€.\
            \nAre you sure you want to continue? [Y/n] ", )
        if resp.lower()=="n":
            exit()
        elif resp!="" and resp.lower()!="y":
            raise ValueError("Please respond with y or n.")

        
        deepl_translate.translate_tasks(tasks_to_translate=args.tasks,
                                        split=args.split,
                                        limit=args.limit,
                                        sample=args.sample,
                                        langs=args.lang,
                                        dummy=args.dummy,
                                        out_dir=args.out_dir,
                                        compute_costs=False)

if __name__=="__main__":
    main()
