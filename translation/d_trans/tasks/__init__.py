from . import arc
from . import hellaswag
from . import truthfulqa
from . import mmlu
from . import winogrande
from . import gsm8k
from . import instruction

task_registry = {
    **arc.generate_tasks(),
    "hellaswag": hellaswag.Hellaswag,
    "truthfulqa_mc": truthfulqa.TruthfulQA_mc,
    "truthfulqa_gen": truthfulqa.TruthfulQA_gen,
    "gsm8k": gsm8k.GSM8k,
    **mmlu.generate_tasks(),
    'instruction': instruction.instructions,
}