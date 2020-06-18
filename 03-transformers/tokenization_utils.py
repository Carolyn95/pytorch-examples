# _*_ coding: utf-8 _*_
# __author__ = 'Carolyn CHEN'

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import json
import six
from io import open
# from .file_utils import cached_path

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'

class PreTrainedTokenizer(object):
    """
    Base class for all tokenizers.
    Including all shared methods for tokenization and special tokens, methods downloading/caching/loading pretrained tokenizers, adding tokens to the vocabulary.
    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the specific vocabulary augmentation methos of the various underlying dictionary structures (BPE, sentencepiece...).

    Attributes (can be overridden by derived classes):
     - vocab_files_names: a python `dict` object, as keys, the `__init__` keyword name of each vocabulary file required by the model, as values, the filename for saving the associated file (string).
     - pretrained_vocab_files_map: a python `dict of dict` object, the high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the associated pretrained vocabulary file.
     - max_model_input_sizes: a python `dict` with, as keys, the `short-cut-names` of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model or None if the model has no maximum input size.

     Parameters:
      - `bos_tokens`: optional, string; a beginning of a sentence token. will be associated to `self.bos_token`
      - `eos_token`: optional, string; an end of a sentence token. will be associated to `self.eos_token` 
      - `unk_token`: optional, string; an unknown token, will be associated to `self.unk_token`
      - `sep_token`: optional, string; a separation token (to separate context and query in an input sequence), will be associated to `self.sep_token`
      - `pad_token`: optional, string; a padding token, will be associated with self.pad_token
      - `cls_token`: optional, string; a classification token (to extract a summary of an input sequence leveraging self-attention along the full depth of the model), will be associated to `self.clf_token`
      - `mask_token`: optional, string; a masking token (when training a model with masked-language modeling), will be associated to `self.mask_token`
      - `additional_special_tokens`: optional, list; a list of additional special tokens, adding all special tokens here ensure they won't be split by the tokenization process. will be associated to `self.additional_special_tokens`
    """
    vocab_file_names = {}
    pretrained_vocab_files_map = {}
    max_model_input_sizes = {}
    SPECIAL_TOKENS_ATTRIBUTES = ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'additianl_special_tokens']

    @property
    def bos_token(self):
        # beginning of sentence token, log an error if used while not having been set
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        # end of sentence token, log an error if used while not having been set
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def unk_token(self):
        # unknown token, log an error if used while not having been set
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet")
        return self._unk_token

    @property
    def sep_token(self):
        # separation token, separate context and query in an input sequence, log an erro if used while not having been set
        if self._sep_token is None:
            logger.error("Using sep_token, but it is not set yet")
        return self._sep_token

    @property
    def pad_token(self):
        # padding token, log an error if used while not having been set
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet")
        return self._pad_token

    @property
    def cls_token(self):
        # classification token, to extract a summary of an input sequence leveraging self-attention along the full depth of the model. log an error if used while not having been set
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet")
        return self._cls_token

    @property
    def mask_token(self):
        # mask token, when training a model with masked-language modeling. log an error if used while not having been set
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet")
        return self._mask_token

    @property
    def additional_special_tokens(self):
        # all the additional special tokens you may want to use (list of strings, log an error if used while not having been set
        if self._addtional_special_tokens is None:
            logger.error("Using additional_special_tokens, but it is not set yet")
        return self._additional_spectial_tokens

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_toke = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    def __init__(self, max_len=None, **kwargs):
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []
        
        self.max_len = max_len if max_len is not None else int(1e12)
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(value, (list, tuple)) and all(isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
                else:
                    assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
                setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        r"""
        Instantiate a class `pytorch_transformers.PreTrainedTokenizer` (or a derived class) from a pretrained tokenizer.

        Args:
            pretrained_model_name_or_path: either:
            - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, :wq

        """
