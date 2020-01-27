from allennlp.data.fields import TextField
from allennlp.data import Token, TokenIndexer
from overrides import overrides
from allennlp.data import Vocabulary
from allennlp.data.token_indexers.token_indexer import TokenIndexer, TokenType
from allennlp.data.token_indexers import SingleIdTokenIndexer
from copy import copy

from typing import List, Dict
TokenList = List[TokenType]


class SourceTextField(TextField):
    def __init__(self, tokens: List[Token], token_indexers: Dict[str, TokenIndexer]) -> None:
        assert(len(token_indexers)==1), "Only one indexer is allowed in a SourceTextField"
        super().__init__(tokens, token_indexers)

    @overrides
    def index(self, vocab: Vocabulary):
        pass

    def index(self, vocab: Vocabulary) -> List[str]:
        token_arrays: Dict[str, TokenList] = {}
        indexer_name_to_indexed_token: Dict[str, List[str]] = {}
        token_index_to_indexer_name: Dict[str, str] = {}

        for indexer_name, indexer in self._token_indexers.items():
            assert type(indexer)==SingleIdTokenIndexer, "The indexer must be a singleidtokenindexer"
            token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)

            oovs_list : List[str] = []
            for key, val in token_indices.items():
                #key is string, val is array of ints
                oov_id = vocab._token_to_index[indexer.namespace][vocab._oov_token]

                ids_with_unks : List[int] = val
                ids_with_oovs : List[int] = []

                for _id, word in zip(ids_with_unks, self.tokens):
                    if _id == oov_id:
                        if word.text not in oovs_list:
                            oovs_list.append(word.text)
                        ids_with_oovs.append(vocab.get_vocab_size(indexer.namespace) + oovs_list.index(word.text))
                    else:
                        ids_with_oovs.append(_id)

                token_arrays.update({
                    "ids_with_unks": ids_with_unks,
                    "ids_with_oovs": ids_with_oovs,
                    "num_oovs": [len(oovs_list)]
                })
            indexer_name_to_indexed_token[indexer_name] = ["ids_with_unks", "ids_with_oovs", "num_oovs"]
            token_index_to_indexer_name["ids_with_unks"] = indexer_name
            token_index_to_indexer_name["ids_with_oovs"] = indexer_name
            token_index_to_indexer_name["num_oovs"] = indexer_name


        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name
        self._oovs = oovs_list

        return self._oovs

class TargetTextField(TextField):
    def __init__(self, tokens: List[Token], token_indexers: Dict[str, TokenIndexer]) -> None:
        super().__init__(tokens, token_indexers)

    @overrides
    def index(self, vocab: Vocabulary, oovs_list: TokenList):
        token_arrays: Dict[str, TokenList] = {}
        indexer_name_to_indexed_token: Dict[str, List[str]] = {}
        token_index_to_indexer_name: Dict[str, str] = {}
        for indexer_name, indexer in self._token_indexers.items():
            assert type(indexer)==SingleIdTokenIndexer, "The indexer must be a singleidtokenindexer"
            token_indices = indexer.tokens_to_indices(self.tokens, vocab, indexer_name)

            for key, val in token_indices.items():
                oov_id = vocab._token_to_index[indexer.namespace][vocab._oov_token]

                ids_with_unks : List[int] = val
                ids_with_oovs : List[int] = []

                for _id, word in zip(ids_with_unks, self.tokens):
                    if _id == oov_id:
                        if word.text not in oovs_list:
                            ids_with_oovs.append(_id)  # let it be the vocab id for OOV
                        else:
                            ids_with_oovs.append(vocab.get_vocab_size(indexer.namespace) + oovs_list.index(word.text))
                    else:
                        ids_with_oovs.append(_id)

                token_arrays.update({
                    "ids_with_unks": ids_with_unks,
                    "ids_with_oovs": ids_with_oovs
                })

            indexer_name_to_indexed_token[indexer_name] = ["ids_with_unks", "ids_with_oovs"]
            token_index_to_indexer_name["ids_with_unks"] = indexer_name
            token_index_to_indexer_name["ids_with_oovs"] = indexer_name

        self._indexed_tokens = token_arrays
        self._indexer_name_to_indexed_token = indexer_name_to_indexed_token
        self._token_index_to_indexer_name = token_index_to_indexer_name
        self._oovs = oovs_list
