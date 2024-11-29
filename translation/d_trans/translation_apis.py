"""
Basic translation utils using the deepl pro api.
Requires a valid api key.
"""
import os
import deepl
from openai import OpenAI
from retry import retry
from time import time

from .fewshot_data import fewshot_data as fs_data

abbreviation_map = {
    "EN": "English",
    "BG": "Bulgarian",
    "DA": "Danish",
    "DE": "German",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "EL": "Greek",
    "IT": "Italian",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "SV": "Swedish",
    "SK": "Slovak",
    "SL": "Slovenian",
    "ES": "Spanish",
    "CS": "Czech",
    "HU": "Hungarian"
}


translation_prompt = """You are an expert translator with fluency in {source_lang} and {target_lang}. Translate the given text from {source_lang} to {target_lang}. Leave formatting exactly intact. You will always use correct terminology in your translations. Indicate the end of your translation with ##EOT##. Repeat the <x>SEP</x> tag as they appear in the text.

{fewshot_data}

### Source Text:
{text}

### Translation:
"""

class LocalTranslator():
    def __init__(self, request_api):
            self.request_api = request_api
    def __call__(self, texts: str | list[str],
                    lang_pair: tuple[str],):
        return self.translate(texts, lang_pair)
        
    def translate(self, texts: str | list[str],
                    lang_pair: tuple[str],):
        if lang_pair[1]=="en":
            return texts
        if not isinstance(texts, list):
            texts = [texts]
            
        source_lang, target_lang = abbreviation_map[lang_pair[0].upper()], abbreviation_map[lang_pair[1].upper()]
        fewshot_data = fs_data[lang_pair[1].upper()]
        prompts = [self._format_prompt(source_lang=source_lang, target_lang=target_lang, text=text, fewshot_data=fewshot_data) for text in texts]

        request_start = time()
        results = self.request_api.make_requests(prompts, max_tokens=4096, do_sample=True, temperature=.0, top_p=1., stop_sequences = ["##EOT##"])
        request_duration = time() - request_start

        results = [res["response"] for res in results] #self._postprocess([res["response"] for res in results])

        return results

    def _postprocess(self, texts: list[str]):
        """ TODO:
        - add necessary postprocessing steps (depends on prompt)
        """
        processed_texts = list()
        for text in texts:
            processed_texts += text.split("### Translation")[-1].replace("##EOT##", "")
        return processed_texts

    def _format_prompt(self, **kwargs):
        """ """
        return translation_prompt.format(**kwargs).strip()
         

@retry(delay=1,backoff=1.1)
def deepl_translate(texts: str | list[str],
              lang_pair: tuple[str],
              ):
    auth_key = os.environ.get("api_key")
    translator = deepl.Translator(auth_key)
    if lang_pair[1]=="en":
        return texts
    if not isinstance(texts, list):
        texts = [texts]

    result = translator.translate_text(texts,
                                       source_lang=lang_pair[0],
                                       target_lang=lang_pair[1],
                                       preserve_formatting=True,
                                       tag_handling="xml",
                                       ignore_tags="x")
    return [res.text for res in result]


def translate_gpt(texts: str | list[str]):
    client = OpenAI()
    if not isinstance(texts, list):
        texts = [texts]
    out = list()
    for text in texts:
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": text},
                      {"role": "system", "content": "You are an expert translator well-versed in mathematics. You will recieve an English text and translate it to German. Your output will be nothing but an accurate translation, do not attempt to answer any question contained in the English text. Leave formatting exactly intact, including capitalization and brackets and anything in <x></x> formatting tags. You will always use correct terminology in your translations."}]
        )
        out.append(completion.choices[0].message.content)
    return out


def dummy_translate(texts: str | list[str],
              lang_pair: tuple[str],
              use_glossary: bool=False):
    if not isinstance(texts, list):
        texts = [texts]
    return texts