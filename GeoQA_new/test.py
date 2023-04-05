# import torch
# from torch import tensor
#
# if __name__=='__main__':
#     a = torch.ones((2, 3))
#     print(a)
#     a1 = torch.sum(a)
#     a2 = torch.sum(a, dim=0)
#     a3 = torch.sum(a, dim=1)
#     a4 = torch.sum(a, dim=-1)
#
#     print((a4==0)).unsqueeze(1).unsqueeze(2)

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedBertIndexer

print(PretrainedBertIndexer("bert-base-uncased"))