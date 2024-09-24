import argparse
from copy import deepcopy
import logging
from pathlib import Path
import time
import psutil
import numpy as np
from wordllama import WordLlama
from xklb.utils import printing, objects
from xklb.utils.log_utils import log

logger = logging.getLogger('kmeans_logger')
logger.setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description="Process sentences from a file.")
parser.add_argument('-n', type=int, default=5, help='Number of runs to perform per variable.')
parser.add_argument("--verbose", "-v", action="count", default=0)
parser.add_argument('path', type=str, help='Path to the text file containing sentences.')
args = parser.parse_args()

sentence_strings = Path(args.path).read_text().splitlines()
num_runs = args.n

def contiguous_labels_score(labels):
    if not labels:
        return 0.0

    non_contiguous_count = 0
    current_label = None
    for label in labels:
        if label != current_label:
            non_contiguous_count += 1
            current_label = label

    contiguous_score = 1.0 - (non_contiguous_count - len(set(labels))) / len(labels)
    return contiguous_score


def evaluate_combination(num_clusters, min_iter):
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    wl = WordLlama.load(config='l3_supercat')

    labels, inertia = wl.cluster(
        sentence_strings,
        k=num_clusters,
        n_init=min_iter,
        min_iterations=min_iter,
        max_iterations=min_iter * 2,
        tolerance=1e-3,
        random_state=np.random.RandomState(0),
    )

    end_time = time.time()
    final_memory = process.memory_info().rss

    speed = end_time - start_time
    ram_usage = final_memory - initial_memory
    log.info(labels[:20])
    label_quality = contiguous_labels_score(labels)

    return inertia, speed, ram_usage, label_quality


results = []
for kwargs in objects.product(
    num_clusters=np.linspace(2, int(len(sentence_strings) ** 0.5) * 2, num_runs, dtype=int),
    min_iter=np.linspace(2, 100, num_runs, dtype=int),
):
    inertia, speed, ram_usage, label_quality = evaluate_combination(**kwargs)
    results.append(
        {
            **kwargs,
            'inertia': inertia,
            'speed': speed,
            'ram_usage': ram_usage,
            'label_quality': label_quality,
        }
    )

tbl = deepcopy(results)
tbl = printing.col_filesize(tbl, 'ram_usage')
printing.table(tbl)

best_combination = max(results, key=lambda x: (x['label_quality'], -x['inertia'], -x['ram_usage'], -x['speed']))
print("Best Combination:", best_combination)
