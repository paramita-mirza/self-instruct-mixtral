# Installation
Run `pip install -e .`.
Enter your deepl api key by running:
`export api_key=<your_key_here>`.

# Usage
Run 

```
python main.py --tasks "list,of,tasks" \
               --lang "lang,codes" \
               --out_dir OUT_DIR \
               [--limit LIMIT] \
               [--split SPLIT] \
               [--dummy]
```
to translate tasks. The output will be a `.jsonl`-file containing a document per line, as in the original task.
The dummy option disables actual translation and simply returns the input string at each translation call. This may be helpful to check whethera translation task is correctly implemented.

# Adding tasks
Copy and modify `d_trans/tasks/template.py` to suit the evaluation task in question.
Add the task to `task_registry` in `d_trans/tasks/__init__.py`.
Execute the above command to create translations.
Optionally you may find it helpful to create a dataset builder script for `huggingface datesets`, a template is provided in `d_trans/builder_template.py`.

# Methodology

Dataset instances are translated using deepl Pro.
The main idea is to generate context-sensitive translations of the different components of an instance so as to preserve grammatical and semantic features between them, whereas separately translating each component may not preserve such features in the same way. For example, take the following example from `hellaswag`:

```
'ctx': 'A cartoon animation video is shown with people wandering around and rockets being shot. two men',
'endings':
 ['fight robots of evil and ends with a to be continued.',
  'are then shown in closeups shooting a shot put.',
  'push a child in a speedboat in the water.',
  'look in the cameraman's eye and smile.'],
```

Our translation methodology results in the following:
```
'ctx': 'Es wird ein Zeichentrickfilm gezeigt, in dem Menschen umherlaufen und Raketen abgeschossen werden. zwei Männer',
'endings':
 ['kämpfen gegen Roboter des Bösen und enden mit einem "to be continued"',
  'werden dann in Großaufnahme beim Kugelstoßen gezeigt.',
  'schieben ein Kind in einem Schnellboot im Wasser.',
  'schauen dem Kameramann in die Augen und lächeln.'],
```

Note how the "look" in the last possible sentence completion is properly translated to the 3rd person plural. Individual translation of this segment may translate "look" into imperative or infinitive, losing the semantics.

We implement this by making use of deepl's xml-handling features (https://www.deepl.com/docs-api/xml). Setting `<x>` as a tag to be ignored, we piece together the instance components separated by `<x>SEP</x>`, translate and then split again on the guaranteed-to-be-unaltered `<x>SEP</x>` token.