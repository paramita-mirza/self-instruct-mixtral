import os

import datasets
import json

_CITATION = """
"""

_DESCRIPTION = """
"""

LANGS = "de,fr,it,es".split(
    ","
)


class TaskConfig(datasets.BuilderConfig):
    def __init__(self, lang, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        #TODO: adjust the following
        self.name = lang
        self.test_url = f"task_name_{lang}_test.jsonl"        
        self.dev_url = f"task_name_{lang}_validation.jsonl"
        self.train_url = f"task_name_{lang}_train.jsonl"


class Task(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [TaskConfig(lang) for lang in LANGS]

    def _info(self):
        #TODO (optional): specify features
        # datasets.Features(
        #     {
        #         "id": datasets.Value("string"),
        #         "title": datasets.Value("string"),
        #         "context": datasets.Value("string"),
        #         "question": datasets.Value("string"),
        #         "answers": datasets.Sequence(
        #             {
        #                 "text": datasets.Value("string"),
        #                 "answer_start": datasets.Value("int32"),
        #             }
        #         ),
        #     }
        # )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            #features=features,
            homepage="",
            license="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        _URL = "https://huggingface.co/datasets/task_name/resolve/main/"       #TODO: adjust URLs
        urls_to_download = {
            "test": _URL + self.config.test_url,
            "dev": _URL + self.config.dev_url,
            "train": _URL + self.config.dev_url,
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [    #TODO: adjust generators (e.g. delete train generator)
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"],
                },
            ),
        ]

    def _generate_examples(self, filepath):
        data = list()
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        for i, instance in enumerate(data):
            yield i, instance
