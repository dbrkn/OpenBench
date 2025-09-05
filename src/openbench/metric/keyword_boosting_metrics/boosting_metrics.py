import os
import json
from typing import List, Dict, Any
import texterrors
import logging
from pathlib import Path

from pyannote.metrics.base import BaseMetric
from ..registry import MetricRegistry
from ..metric import MetricOptions
from ...types import PipelineType
from ...pipeline_prediction import Transcript





class BaseKeywordMetric(BaseMetric):
    """Base class for keyword boosting metrics."""
    
    def __init__(self):
        """Initialize keyword metric."""
        super().__init__()
        # Use Whisper's BasicTextNormalizer
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        self.text_normalizer = BasicTextNormalizer()

    def compute_keyword_stats(
        self, reference: Transcript, hypothesis: Transcript, **kwargs
    ) -> Dict[str, Any]:
        """Compute keyword statistics between reference and hypothesis."""
        # Get keywords from kwargs (passed by the metric framework)
        keywords = kwargs.get('dictionary', [])
        
        # Convert transcripts to text
        ref_text = " ".join([word.word for word in reference.words])
        hyp_text = " ".join([word.word for word in hypothesis.words])
        
        # Apply normalization to hypothesis only
        hyp_text = self.text_normalizer(hyp_text)
        
        # Normalize keywords as well
        normalized_keywords = [self.text_normalizer(kw) for kw in keywords]
        
        # Get alignment using texterrors
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        texterrors_ali = texterrors.align_texts(ref_words, hyp_words, False)
        
        # Create alignment pairs
        ali = []
        for i in range(len(texterrors_ali[0])):
            ali.append((texterrors_ali[0][i], texterrors_ali[1][i]))

        # Compute max ngram order
        max_ngram_order = max([len(item.split()) for item in normalized_keywords])
        key_words_stat = {}
        for word in normalized_keywords:
            key_words_stat[word] = [0, 0, 0]  # [tp, gt, fp]

        eps = "<eps>"

        # 1-grams
        for idx in range(len(ali)):
            word_ref = ali[idx][0]
            word_hyp = ali[idx][1]
            if word_ref in key_words_stat:
                key_words_stat[word_ref][1] += 1  # add to gt
                if word_ref == word_hyp:
                    key_words_stat[word_ref][0] += 1  # add to tp
            elif word_hyp in key_words_stat:
                key_words_stat[word_hyp][2] += 1  # add to fp

        # 2-grams and higher
        for ngram_order in range(2, max_ngram_order + 1):
            # For reference phrase
            idx = 0
            item_ref = []
            while idx < len(ali):
                if item_ref:
                    item_ref = [item_ref[1]]
                    idx = item_ref[0][1] + 1
                while len(item_ref) != ngram_order and idx < len(ali):
                    word = ali[idx][0]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_ref.append((word, idx - 1))
                if len(item_ref) == ngram_order:
                    phrase_ref = " ".join([item[0] for item in item_ref])
                    phrase_hyp = " ".join([ali[item[1]][1] for item in item_ref])
                    if phrase_ref in key_words_stat:
                        key_words_stat[phrase_ref][1] += 1  # add to gt
                        if phrase_ref == phrase_hyp:
                            key_words_stat[phrase_ref][0] += 1  # add to tp

            # For false positive hypothesis phrase
            idx = 0
            item_hyp = []
            while idx < len(ali):
                if item_hyp:
                    item_hyp = [item_hyp[1]]
                    idx = item_hyp[0][1] + 1
                while len(item_hyp) != ngram_order and idx < len(ali):
                    word = ali[idx][1]
                    idx += 1
                    if word == eps:
                        continue
                    else:
                        item_hyp.append((word, idx - 1))
                if len(item_hyp) == ngram_order:
                    phrase_hyp = " ".join([item[0] for item in item_hyp])
                    phrase_ref = " ".join([ali[item[1]][0] for item in item_hyp])
                    if phrase_hyp in key_words_stat and phrase_hyp != phrase_ref:
                        key_words_stat[phrase_hyp][2] += 1  # add to fp

        # Compute totals
        tp = sum([key_words_stat[x][0] for x in key_words_stat])
        gt = sum([key_words_stat[x][1] for x in key_words_stat])
        fp = sum([key_words_stat[x][2] for x in key_words_stat])

        return {
            "true_positives": tp,
            "ground_truth": gt,
            "false_positives": fp,
            "keyword_stats": key_words_stat
        }


@MetricRegistry.register_metric(PipelineType.BOOSTING_TRANSCRIPTION, MetricOptions.KEYWORD_FSCORE)
class KeywordFScore(BaseKeywordMetric):
    """Keyword F-Score metric for boosting transcription evaluation."""
    
    @classmethod
    def metric_name(cls) -> str:
        return "keyword_fscore"
    
    @classmethod
    def metric_components(cls) -> list[str]:
        return [
            "true_positives",
            "ground_truth", 
            "false_positives"
        ]
    
    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> Dict[str, int]:
        """Compute keyword F-score components."""
        stats = self.compute_keyword_stats(reference, hypothesis, **kwargs)
        return {
            "true_positives": stats["true_positives"],
            "ground_truth": stats["ground_truth"],
            "false_positives": stats["false_positives"]
        }
    
    def compute_metric(self, detail: Dict[str, int]) -> float:
        """Compute F-score from components."""
        tp = detail["true_positives"]
        gt = detail["ground_truth"] 
        fp = detail["false_positives"]
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (gt + 1e-8)
        fscore = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return fscore


@MetricRegistry.register_metric(PipelineType.BOOSTING_TRANSCRIPTION, MetricOptions.KEYWORD_PRECISION)
class KeywordPrecision(BaseKeywordMetric):
    """Keyword Precision metric for boosting transcription evaluation."""
    
    @classmethod
    def metric_name(cls) -> str:
        return "keyword_precision"
    
    @classmethod  
    def metric_components(cls) -> list[str]:
        return [
            "true_positives",
            "false_positives"
        ]
    
    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> Dict[str, int]:
        """Compute keyword precision components."""
        stats = self.compute_keyword_stats(reference, hypothesis, **kwargs)
        return {
            "true_positives": stats["true_positives"],
            "false_positives": stats["false_positives"]
        }
    
    def compute_metric(self, detail: Dict[str, int]) -> float:
        """Compute precision from components."""
        tp = detail["true_positives"]
        fp = detail["false_positives"]
        
        return tp / (tp + fp + 1e-8)


@MetricRegistry.register_metric(PipelineType.BOOSTING_TRANSCRIPTION, MetricOptions.KEYWORD_RECALL)
class KeywordRecall(BaseKeywordMetric):
    """Keyword Recall metric for boosting transcription evaluation."""
    
    @classmethod
    def metric_name(cls) -> str:
        return "keyword_recall"

    @classmethod
    def metric_components(cls) -> list[str]:
        return [
            "true_positives",
            "ground_truth"
        ]

    def compute_components(self, reference: Transcript, hypothesis: Transcript, **kwargs) -> Dict[str, int]:
        """Compute keyword recall components."""
        stats = self.compute_keyword_stats(reference, hypothesis, **kwargs)
        return {
            "true_positives": stats["true_positives"],
            "ground_truth": stats["ground_truth"]
        }

    def compute_metric(self, detail: Dict[str, int]) -> float:
        """Compute recall from components."""
        tp = detail["true_positives"]
        gt = detail["ground_truth"]

        return tp / (gt + 1e-8)