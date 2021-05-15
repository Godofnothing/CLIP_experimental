import torch

from .tokenizer import SimpleTokenizer

class TextTransformer:

  def __init__(self, tokenizer, context_length):
    self.tokenizer = tokenizer
    self.context_length = context_length      

  def transform(self, class_label):
    text_input = f"This is a photo of {class_label}"
    tokens = self.tokenizer.encode(text_input)

    text_item = torch.zeros(self.context_length, dtype=torch.long)
    sot_token = self.tokenizer.encoder['<|startoftext|>']
    eot_token = self.tokenizer.encoder['<|endoftext|>']

    tokens = [sot_token] + tokens + [eot_token]
    text_item[:len(tokens)] = torch.tensor(tokens)

    return text_item