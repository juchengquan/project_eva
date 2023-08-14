"""
Source: https://github.com/huggingface/datasets/blob/main/metrics/wer/wer.py
"""
# Copyright 2021 The HuggingFace Datasets Authors.
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
""" Word Error Ratio (WER) metric. """

from typing import List
from jiwer import compute_measures

class WER:
    def compute(self, 
        predictions: List[str]=None, 
        references: List[str]=None,
        concatenate_texts: bool=False,
    ):
        if concatenate_texts:
            return compute_measures(references, predictions)["wer"]
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(predictions, references):
                measures = compute_measures(reference, prediction)
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total += measures["substitutions"] + measures["deletions"] + measures["hits"]
            return incorrect / total
