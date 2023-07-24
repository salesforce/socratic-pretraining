"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import math
from concurrent.futures.process import ProcessPoolExecutor

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[1:] = input_ids[:-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MultiEncoderDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        max_num_chunks: int,
        max_target_length: int,
        stride: bool = False,
        pad: bool = True,
        num_samples: int = None,
        verbose: bool = False,
        ignore_pad_token_for_loss: bool = True,
        max_workers=1,
        mode_token=None,
        provided_decoder_input=False,
    ):

        self.tokenizer = tokenizer
        self.chunk_tokenizer = ChunkTokenizer(
            tokenizer,
            chunk_size,
            max_num_chunks,
            stride,
            pad
        )
        self.mode_token = mode_token
        self.provided_decoder_input = provided_decoder_input
        self.max_target_length = max_target_length
        self.num_samples = num_samples
        self.verbose = verbose
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self._encode_data(data_path, max_workers)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index]

    def _encode_data(self, file_path, max_workers):
        with open(file_path) as f:
            if max_workers == 1:
                encodings = list(map(self._process_line, enumerate(f)))
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    encodings = executor.map(self._process_line, enumerate(f))
        self.encodings = [enc for enc in encodings if enc is not False]
        if self.num_samples is not None:
            assert self.num_samples == len(self.encodings)

    def _process_line(self, index_line):
        i, line = index_line
        if i % 100 == 0:
            print('Processed', i, 'records', datetime.datetime.now())
        if self.num_samples is not None and i >= self.num_samples:
            return False
        data = json.loads(line)
        source = data['source']
        query = data.get('query')
        target = data['target']
        encoding = self._encode_example(
            source,
            target,
            query
        )
        if self.verbose and i == 0:
            print('First record in dataset:')
            for token_ids in encoding['input_ids']:
                print()
                print(self.tokenizer.decode(token_ids))
        return encoding

    def _encode_example(self, source, target, query=None):

        output = self.chunk_tokenizer(source, query, mode=self.mode_token)
        source_ids = output['input_ids']
        source_attention_mask = output['attention_mask']

        tokenized_answer = self.tokenizer(
            target,
            pad_to_max_length=True,
            max_length=self.max_target_length,
            return_tensors="pt",
            truncation=True
        )
        target_ids = tokenized_answer['input_ids'].squeeze()
        if self.ignore_pad_token_for_loss:
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        out = {
            'input_ids': source_ids,
            'attention_mask': source_attention_mask,
            'labels': target_ids,
            'decoder_attention_mask': tokenized_answer['attention_mask'].squeeze(),
        }

        if self.provided_decoder_input:
            bos_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.special_tokens_map['bos_token']
            )
            decoder_input_ids = shift_tokens_right(
                target_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=bos_token_id
            )
            qsep_in_target = target_ids == self.tokenizer.convert_tokens_to_ids('<qsep>')
            if qsep_in_target.sum() > 0:
                question_mask_id = ((qsep_in_target).nonzero(as_tuple=True)[0][0])
                target_ids[:question_mask_id] = -100
            out['decoder_input_ids'] = decoder_input_ids
            out['labels'] = target_ids
        return out


class ChunkTokenizer:
    """Chunks and tokenizes input text and optional query for input to multi-encoder model. Does both chunking and
    tokenizing because the chunking is based on tokenized text."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        max_num_chunks: int,
        stride: bool = False,
        pad: bool = False
    ):
        """
        Args:
            tokenizer: tokenizer used to tokenize text
            chunk_size: chunk size in number of tokens
            max_num_chunks: maximum number of chunks in total (optional)
            stride: whether to use striding
            pad: whether to "pad" chunks with empty strings to attain max_num_chunks chunks
        """
        if pad and not max_num_chunks:
            raise ValueError("Cannot pad without specifying max_num_chunks")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        self.stride = stride
        self.pad = pad

    def __call__(
        self,
        source: str,
        query: str = None,
        mode: str = None,
    ):
        """
        Args:
            source: source text
            query: optional query text
            mode: optional mode token
        Returns:
            dictionary with tokenized chunks
        """
        if query and mode:
            prefix = f"<s>{mode}{query}<qsep>"
        elif mode:
            prefix = f"<s>{mode}"
        elif query:
            prefix = f"<s>{query}<qsep>"
        else:
            prefix = f"<s>"
        prefix_tokens = self.tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.chunk_size,
            truncation=True
        )['input_ids']
        prefix_len = prefix_tokens.size(-1)

        suffix_chunk_size = self.chunk_size - prefix_len
        chunk_input_ids_all = []
        chunk_attention_mask_all = []

        suffix = f"{source}</s>"
        suffix_total_size = self.max_num_chunks * suffix_chunk_size
        input_ids = self.tokenizer(
            suffix,
            add_special_tokens=False,
            truncation=True,
            max_length=suffix_total_size,
        )['input_ids']

        if self.stride and self.max_num_chunks > 1:
            use_offset_list = [False, True]
        else:
            use_offset_list = [False]
        for use_offset in use_offset_list:
            if use_offset:
                offset = math.floor(suffix_chunk_size / 2)
                num_chunks = self.max_num_chunks - 1
                suffix_tokens = input_ids[offset: offset + num_chunks * suffix_chunk_size]
            else:
                suffix_tokens = input_ids[:suffix_total_size]
                num_chunks = self.max_num_chunks

            suffix_attention = [1] * len(suffix_tokens)
            if self.pad:  # If padding chunks to num chunks, need to fill out suffix_total_size
                pad_length = max(num_chunks * suffix_chunk_size - len(suffix_tokens), 0)
            else:  # Pad to next multiple of chunk size
                remainder = len(suffix_tokens) % suffix_chunk_size
                if remainder == 0:
                    pad_length = 0
                else:
                    pad_length = suffix_chunk_size - remainder
            suffix_tokens += [self.tokenizer.pad_token_id] * pad_length
            suffix_attention += [0] * pad_length

            suffix_tokens = torch.tensor(suffix_tokens)
            suffix_attention = torch.tensor(suffix_attention)

            suffix_chunks = suffix_tokens.view(-1, suffix_chunk_size)
            suffix_attention = suffix_attention.view(-1, suffix_chunk_size)

            prefix_chunks = prefix_tokens.expand(suffix_chunks.size(0), -1)
            prefix_attention = torch.ones_like(prefix_chunks)

            chunk_input_ids = torch.cat((prefix_chunks, suffix_chunks), dim=1)
            chunk_attention_mask = torch.cat((prefix_attention, suffix_attention), dim=1)

            chunk_input_ids_all.append(chunk_input_ids)
            chunk_attention_mask_all.append(chunk_attention_mask)

        return {
            "input_ids": torch.cat(chunk_input_ids_all, dim=0),
            "attention_mask": torch.cat(chunk_attention_mask_all, dim=0)
        }
