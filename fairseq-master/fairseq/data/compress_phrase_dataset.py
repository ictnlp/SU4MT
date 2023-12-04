# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
import json
import pdb
import math

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths},
        "target": target,
        "mono": True,
        "pad_idx": pad_idx,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    return batch


class CompressPhraseDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        left_pad_source=True,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        pad_to_multiple=1,
        # dynamic_mask_ratio=0.35,
        # phrase_data=None,
    ):
    # 第一是读入phrase表，或抽phrase，或别的方式，标注为P，保留P的内容作为phrase encoder输入，mask P的内容作为一般encoder输入；
    # 第二是统计mask P之后所有mask占总token的数量如果小于35%，就继续mask直到35%，标注为D，保持两边输入的D(ynamic mask)是一致的。
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.sizes = self.src_sizes
        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        # if phrase_data is not None:
        #     self.phrase_data = phrase_data
        # else:
        #     self.phrase_data = {}
        # self.dynamic_mask_ratio = dynamic_mask_ratio

    def get_batch_shapes(self):
        return self.buckets
#    def __getitem__(self, index):
#         tgt_item = self.tgt[index] if self.tgt is not None else None
#         src_item = self.src[index]
#         # Append EOS to end of tgt sentence if it does not have an EOS and remove
#         # EOS from end of src sentence if it exists. This is useful when we use
#         # use existing datasets for opposite directions i.e., when we want to
#         # use tgt_dataset as src_dataset and vice versa
#         if self.append_eos_to_target:
#             eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
#             if self.tgt and self.tgt[index][-1] != eos:
#                 tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

#         if self.append_bos:
#             print("Warning: bos is used as [cls] token in phrase embedder fusion module, append_bos is likely to cause unpredictable errors!")
#             bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
#             if self.tgt and self.tgt[index][0] != bos:
#                 tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

#             bos = self.src_dict.bos()
#             if self.src[index][0] != bos:
#                 src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

#         if self.remove_eos_from_source:
#             eos = self.src_dict.eos()
#             if self.src[index][-1] == eos:
#                 src_item = self.src[index][:-1]
        
#         if index in self.phrase_data:
#             phrase_spans = self.phrase_data[index]
#         else:
#             print('AAAAAAindex:',index)
#             phrase_spans = None
#         if self.tgt is None:
#             with torch.no_grad():
#                 # 就是单语任务
#                 # pdb.set_trace()
#                 total_length = len(src_item)
#                 # 保留phrase部分，但是需要动态mask 
#                 phrase_encoder_input = src_item
#                 # mask最多的，phrase和动态mask
#                 encoder_input = src_item.clone().detach()
#                 # 保留完整信息，作为ground
#                 target_input = src_item.clone().detach()
#                 span_count = 0 
#                 remained = set([i for i in range(total_length)])
#                 if phrase_spans:
#                     # 如果没有phrase span说明不是训练步骤，就不mask了
#                     for span in phrase_spans:
#                         assert span[1]-span[0] > 1, "需要保证span长度大于1，回去检查抽span的步骤"
#                         span_count += span[1]-span[0]
#                         for i in range(span[0],span[1]):
#                             encoder_input[i] = self.src_dict.pad()
#                             remained.remove(i)
#                     dynamic_mask_num = math.floor(total_length*self.dynamic_mask_ratio) - span_count
#                     # pdb.set_trace()
#                     if dynamic_mask_num > 0:
#                         remained = torch.Tensor(list(remained))
#                         dynamic_mask = torch.multinomial(remained,dynamic_mask_num)
#                         encoder_input.index_fill(0, dynamic_mask, self.src_dict.pad())
#                         phrase_encoder_input.index_fill(0, dynamic_mask, self.src_dict.pad())
#                         # for idx in dynamic_mask:
#                         #     encoder_input[idx] = self.src_dict.pad()
#                         #     phrase_encoder_input[idx] = self.src_dict.pad()
#                         del dynamic_mask
#                 del remained
#                 del span_count
#                 del total_length

#             example = {
#                 "id": index,
#                 "source": encoder_input,
#                 "phrase_source": phrase_encoder_input,
#                 "target": target_input,
#                 "phrase_info": phrase_spans,
#             }
#         else:
#             # 双语任务
#             with torch.no_grad():
#                 phrase_encoder_input = src_item.clone().detach()
#             example = {
#                 "id": index,
#                 "source": src_item,
#                 "phrase_source": phrase_encoder_input,
#                 "target": tgt_item,
#                 "phrase_info": phrase_spans,
#             }
#         return example

    def __getitem__(self, index):
        tokens = self.src[index]
        source, target = tokens, tokens.clone()
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa

        if self.append_bos:
            print("Warning: bos is used as [cls] token in phrase embedder fusion module, append_bos is likely to cause unpredictable errors!")

            bos = self.src_dict.bos()
            if source[0] != bos:
                source = torch.cat([torch.LongTensor([bos]), source])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if source[-1] == eos:
                source = source[:-1]

        return {
            "id": index,
            "source": source,
            "target": target,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.src_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) 

    def prefetch(self, indices):
        self.src.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.src_sizes, indices, max_sizes,
        )
