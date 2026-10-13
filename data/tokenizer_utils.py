from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self._greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
        total_tokens = self.count_tokens(text, vocab)
        score = total_tokens / len(words)
        return round(score, 4)
        
    def _greedy_tokenize(self, s: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(s)
        while i < n:
            match = None
            for j in range(n, i, -1):
                substring = s[i:j]
                if substring in vocab:
                    match = substring
                    break
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(s[i])
                i += 1
        return tokens