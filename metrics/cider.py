"""
implementation of cider score is copied from
https://github.com/huggingface/evaluate/tree/62e96446a2224700d16056625a5105e8fdc5f51c/metrics/CIDEr
"""
from typing import List
import urllib.request
import os
import tempfile
import subprocess

from pycocoevalcap.cider.cider import CiderScorer

_URLS = {
    "stanford-corenlp": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar"
}


def tokenize(tokenizer_path: str, predictions: List[str], references: List[List[str]]):
    PUNCTUATIONS = [
        "''",
        "'",
        "``",
        "`",
        "-LRB-",
        "-RRB-",
        "-LCB-",
        "-RCB-",
        ".",
        "?",
        "!",
        ",",
        ":",
        "-",
        "--",
        "...",
        ";",
    ]

    cmd = [
        "java",
        "-cp",
        tokenizer_path,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    sentences = "\n".join(
        [
            s.replace("\n", " ")
            for s in predictions + [ref for refs in references for ref in refs]
        ]
    )

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sentences.encode())

    cmd.append(f.name)
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
    token_lines = token_lines.decode()
    lines = [
        " ".join([w for w in line.rstrip().split(" ") if w not in PUNCTUATIONS])
        for line in token_lines.split("\n")
    ]

    os.remove(f.name)

    pred_size = len(predictions)
    ref_sizes = [len(ref) for ref in references]

    predictions = lines[:pred_size]
    start = pred_size
    references = []
    for size in ref_sizes:
        references.append(lines[start : start + size])
        start += size

    return predictions, references


def download_and_prepare(save_path):

    jar_path = os.path.join(save_path, "stanford-corenlp-3.4.1.jar")
    urllib.request.urlretrieve(_URLS["stanford-corenlp"], jar_path)
    print(f"tokenizer jar file is saved at {jar_path}")
    return jar_path

def compute(predictions, references, tokenizer_path, n=4, sigma=6.0):
    predications, references = tokenize(
        tokenizer_path, predictions, references
    )
    scorer = CiderScorer(n=n, sigma=sigma)
    for pred, refs in zip(predications, references):
        scorer += (pred, refs)
    score, scores = scorer.compute_score()
    return {"CIDEr": score}
