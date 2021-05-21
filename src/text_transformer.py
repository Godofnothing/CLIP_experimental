from typing import List

import torch


class TextTransformer:

    def __init__(self, tokenizer, templates, context_length):
        """Adds prefix from templates and tokenizes the text

        Args:
            tokenizer: encodes string to array of numbers with .encode() method
            templates (List[str]): Prompts with {} placeholders
            context_length (int): maximum text length to which text is padded with zeros
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.templates = templates

        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']

    def _pad_sentence(self, sent: List[int]) -> List[int]:
        if len(sent) > self.context_length - 2:
            raise ValueError(f'Sentence {sent} is too long')
        else:
            return [self.sot_token] + \
                sent + [0] * (self.context_length - 2 - len(sent)) + \
                [self.eot_token]

    def __call__(self, class_label: str):
        texts = [template.format(class_label) for template in self.templates]
        tokens = [self.tokenizer.encode(text) for text in texts]
        padded = [self._pad_sentence(sent) for sent in tokens]
        return torch.tensor(padded, dtype=torch.long)
