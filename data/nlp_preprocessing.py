import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentence = positive + negative
        vocab = set()
        for sentence in all_sentence:
            for i in sentence.split():
                vocab.add(i)
        sorted_vocab = sorted(list(vocab))
        word_to_id = {word: idx + 1 for idx, word in enumerate(sorted_vocab)}
        encoded_tensors = []
        for sentence in all_sentence:
            words = sentence.split()
            idx = [word_to_id[word] for word in words]
            encoded_tensors.append(torch.tensor(idx, dtype = torch.long))
        padded = nn.utils.rnn.pad_sequence(encoded_tensors, batch_first = True, padding_value = 0)
        return padded.float()
