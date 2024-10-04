import json

instances_file = "../self_instruct_alpaca/data/mixtral_8x22b_generations_alpaca_100/machine_generated_instructions_instances.jsonl"

if __name__ == "__main__":
    print(f"File to evaluate: {instances_file}")
    input_file = instances_file

    with open(input_file, 'r') as json_file:
        json_list = list(json_file)

    output_file = input_file.replace(".jsonl", '') + "_input_checked.jsonl"
    with open(output_file, 'w') as fout:
        for json_str in json_list:
            instance = json.loads(json_str)
            if instance["input"]:
                print("\n## instruction: ", instance["instruction"])
                print("## input: ", instance["input"])
                answer = input("Is input relevant? [yes, no] ")
            else:
                answer = "n/a"

            instance["input_relevance"] = answer
            fout.write(json.dumps(instance) + "\n")