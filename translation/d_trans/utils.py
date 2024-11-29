import fnmatch
import random
import math
import os 

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice

def get_matches(choices,query):
    out = list()
    data_paths = list()
    for q in query.split(","):
        if os.path.isfile(q):
            data_paths += [q]
            out += ['instruction']
        else:
            matches = fnmatch.filter(choices,q)
            data_paths += [None] * len(matches)
            out += matches
    return out, data_paths #list(set(out))

def stratified_sample(limit, docgroups):
    grouplens = [len(group) for group in docgroups]
    total = sum(grouplens)
    props = [el/total for el in grouplens]
    outdocs = list()
    for group, prop in zip(docgroups,props):
        outdocs += random.sample(group,math.floor(limit*prop))
    return outdocs

def group_samples(docs, task):
    topics = {task.get_topic(doc) for doc in docs}
    groups = [list(filter(lambda x: task.get_topic(x)==t, task.get_docs("validation"))) for t in topics]
    return groups