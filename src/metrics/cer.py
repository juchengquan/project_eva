"""
Source: https://github.com/huggingface/datasets/blob/main/metrics/cer/cer.py
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
""" Character Error Ratio (CER) metric. """

from typing import List

import jiwer
import jiwer.transforms as tr
from packaging import version

import platform
if version.parse(platform.python_version()) < version.parse("3.8"):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


SENTENCE_DELIMITER = ""


if version.parse(importlib_metadata.version("jiwer")) < version.parse("2.3.0"):
    class SentencesToListOfCharacters(tr.AbstractTransform):
        def __init__(self, sentence_delimiter: str = " "):
            self.sentence_delimiter = sentence_delimiter

        def process_string(self, s: str):
            return list(s)

        def process_list(self, inp: List[str]):
            chars = []
            for sent_idx, sentence in enumerate(inp):
                chars.extend(self.process_string(sentence))
                if self.sentence_delimiter is not None and self.sentence_delimiter != "" and sent_idx < len(inp) - 1:
                    chars.append(self.sentence_delimiter)
            return chars

    cer_transform = tr.Compose(
        [tr.RemoveMultipleSpaces(), tr.Strip(), SentencesToListOfCharacters(SENTENCE_DELIMITER)]
    )
else:
    cer_transform = tr.Compose(
        [
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
            tr.ReduceToListOfListOfChars(),
        ]
    )

class CER:
    def compute(self, predictions, references, concatenate_texts=False):
        if concatenate_texts:
            return jiwer.compute_measures(
                references,
                predictions,
                truth_transform=cer_transform,
                hypothesis_transform=cer_transform,
            )["wer"]

        incorrect = 0
        total = 0
        for prediction, reference in zip(predictions, references):
            measures = jiwer.compute_measures(
                reference,
                prediction,
                truth_transform=cer_transform,
                hypothesis_transform=cer_transform,
            )
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total += measures["substitutions"] + measures["deletions"] + measures["hits"]

        return incorrect / total
