import json

instances_file = "../self_instruct_alpaca/data/mixtral_8x22b_generations_alpaca_100/machine_generated_instructions_instances_input_checked.jsonl"
openai_file = "../self_instruct_alpaca/data/mixtral_8x22b_generations_alpaca_100/all_generated_output_openai.jsonl"

if __name__ == "__main__":
    print(f"File to evaluate: {instances_file}")
    input_file = instances_file

    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
    with open(openai_file, 'r') as openai_file:
        json_openai = list(openai_file)

    output_file = input_file.replace(".jsonl", '') + "_output_checked.jsonl"
    with open(output_file, 'w') as fout:
        for i, json_str in enumerate(json_list):
            instance = json.loads(json_str)
            openai_output = json.loads(json_openai[i])
            print("\n## instruction: ", instance["instruction"])
            print("## input: ", instance["input"])
            print("## output: ", instance["output"])
            print("## openai output: ", openai_output["openai_output"])
            answer = input("Is output correct? [yes, no] ")

            instance["output_correctness"] = answer
            fout.write(json.dumps(instance) + "\n")