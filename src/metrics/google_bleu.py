"""
https://github.com/huggingface/datasets/blob/main/metrics/google_bleu/google_bleu.py
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
""" Google BLEU (aka GLEU) metric. """

from typing import Dict, List, Union
from nltk.translate import gleu_score

class GoogleBleu:
    def __init__(self):
        pass

    def compute(
        self,
        predictions: List[List[Union[str, int]]],
        references: List[List[List[Union[str, int]]]],
        min_len: int = 1,
        max_len: int = 4,
    ) -> Dict[str, float]:
        return {
            "google_bleu": gleu_score.corpus_gleu(
                list_of_references=references, hypotheses=predictions, min_len=min_len, max_len=max_len
            )
        }
