"""
Source: https://github.com/huggingface/datasets/blob/main/metrics/rouge/rouge.py
"""
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
# import absl  # noqa: F401 # Here to have a nice missing dependency error message early on
# import nltk  # noqa: F401 # Here to have a nice missing dependency error message early on
# import numpy  # noqa: F401 # Here to have a nice missing dependency error message early on
# import six  # noqa: F401 # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring

from typing import List

class Rouge:
    def __init__(self):
        pass

    def compute(self, 
        predictions: List[str], 
        references: List[str], 
        rouge_types: List = None, 
        use_aggregator: bool = True, 
        use_stemmer: bool = False,
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_aggregator:
            result = aggregator.aggregate()
        else:
            result = {}
            for key in scores[0]:
                result[key] = [score[key] for score in scores]

        return result
