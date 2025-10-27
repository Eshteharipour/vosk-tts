"""from https://github.com/keithito/tacotron"""

import re

import torch

from .ru_dictionary import convert
from .symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


def get_bert_embeddings(text, model, tokenizer):
    with torch.no_grad():
        text = text.replace("+", "")
        inputs = tokenizer(text, return_tensors="pt")
        #        print (inputs)
        text_inputs = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        #        print (text_inputs)

        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1).squeeze(0)

        pattern = '[-,.?!;:"]'
        selected = []
        for i, t in enumerate(text_inputs):
            if t[0] != "#" and not re.match(pattern, t):
                #                print (i, t)
                selected.append(i)
        res = res[selected]
        return res


# Kaldi style word position
def get_pos(x):
    if len(x) == 1:
        return [x[0] + "_S"]
    else:
        res = []
        for i, p in enumerate(x):
            if i == 0:
                res.append(p + "_B")
            elif i == len(x) - 1:
                res.append(p + "_E")
            else:
                res.append(p + "_I")
        return res


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result
