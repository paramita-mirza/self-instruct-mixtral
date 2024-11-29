import logging
import io

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

blacklists = {"EN": ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to", "video", "audio", "music", "flowchart", "diagram",],
              "HU": ["kép", "képek", "grafikon", "grafikonok", "kép", "képek", "filé", "fájlok", "mappa", "térképek", "húzni", "telek", "menni", "videó", "hang", "zene", "folyamatábra", "diagram"],
              "DE": ["Bild", "Bilder", "Graph", "Graphen", "Bild", "Bilder", "Datei", "Dateien", "Karte", "Karten", "Zeichnen", "Plotten", "Gehe zu", "Video", "Audio", "Musik", "Flussdiagramm", "Diagramm"],}


language_map = {"BG": {}, 
                    "DA": {},
                    "DE": {"Instructions": "Anweisungen", "Inputs": "Eingaben", "Outputs": "Ausgaben", 
                           "Instruction": "Anweisung", "Input": "Eingabe", "Output": "Ausgabe", 
                           "blacklist": blacklists["DE"]
                           }, 
                    "EN": {"Instructions": "Instructions", "Inputs": "Inputs", "Outputs": "Outputs",
                           "Instruction": "Instruction", "Input": "Input", "Output": "Output", 
                           "blacklist": blacklists["EN"]
                           },
                    "ET": {}, 
                    "FI": {}, 
                    "FR": {}, 
                    "EL": {}, 
                    "IT": {}, 
                    "LV": {}, 
                    "LT": {}, 
                    "NL": {}, 
                    "PL": {},
                    "PT-PT": {}, 
                    "RO": {}, 
                    "SV": {}, 
                    "SK": {}, 
                    "SL": {}, 
                    "ES": {}, 
                    "CS": {}, 
                    "HU": {"Instructions": "Utasítások", "Inputs": "Bemenetek", "Outputs": "Kimenetek", 
                           "Instruction": "Utasítás", "Input": "Bemenet", "Output": "Kimenet", 
                           "blacklist": blacklists["HU"]
                            },
                     }

embedding_models = {"EN": "sentence-transformers/all-roberta-large-v1",
                    "DE": "intfloat/multilingual-e5-large-instruct", # license: MIT
                    "HU": "intfloat/multilingual-e5-large-instruct'" # license: MIT; check language specific models: "SZTAKI-HLT/hubert-base-cc" Sentence transformers: NYTK/sentence-transformers-experimental-hubert-hungarian  
                    } 
