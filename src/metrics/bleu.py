"""
Source: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
"""

# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

from typing import Dict, List, Union

import collections
import math

class Bleu:
    def __init__(self):
        pass

    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.

        Args:
            segment: text segment from which n-grams will be extracted.
            max_order: maximum length in tokens of the n-grams returned by this
                    methods.

        Returns:
            The Counter containing all n-grams upto max_order in segment
            with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts


    def compute(self, 
        predictions: List[List[Union[str, int]]],
        references: List[List[List[Union[str, int]]]],
        max_order: int = 4, 
        smooth: int = False,
    ) -> Dict:
        """Computes BLEU score of translated segments against one or more references.

        Args:
            references: list of lists of references for each translation. Each
                    references should be tokenized into a list of tokens.
            predictions: list of translations to score. Each translation
                    should be tokenized into a list of tokens.
            max_order: Maximum n-gram order to use when computing BLEU score.
            smooth: Whether or not to apply Lin et al. 2004 smoothing.

        Returns:
            3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (reference, translation) in zip(references, predictions):
            reference_length += min(len(r) for r in reference)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for _r in reference:
                merged_ref_ngram_counts |= self._get_ngrams(_r, max_order)
            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1, max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return {
            "bleu": bleu,
            "precisions": precisions,
            "bp": bp,
            "ratio": ratio,
            "translation_length": translation_length,
            "reference_length": reference_length,
        }
