# -*- coding: utf-8 -*-
import typing

import pyarrow as pa
from kiara import KiaraModule
from kiara.data.values import ValueSchema, ValueSet
from spacy.tokens import Doc
from spacy.util import DummyTokenizer


class LemmatizeTokensModule(KiaraModule):
    """Lemmatize a single token list."""

    _module_type_name = "tokens"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        inputs = {"tokens_array": {"type": "list", "doc": "A list of tokens."}}
        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        outputs = {
            "tokens_array": {"type": "list", "doc": "A list of lemmatized tokens."}
        }
        return outputs

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        tokens = inputs.get_value_data("tokens_array")

        # TODO: install this on demand?
        import it_core_news_sm

        it_nlp = it_core_news_sm.load(disable=["tagger", "parser", "ner"])

        lemmatized_doc = []
        for w in tokens:
            w_lemma = [token.lemma_ for token in it_nlp(w)]
            lemmatized_doc.append(w_lemma[0])

        outputs.set_value("tokens_array", lemmatized_doc)


class LemmatizeTokensArrayModule(KiaraModule):
    """Lemmatize an array of token lists.

    Compared to using the ``lemmatize_tokens`` module in combination with ``map``, this is much faster, since it uses
    a spacy [pipe](https://spacy.io/api/language#pipe) under the hood.
    """

    _module_type_name = "tokens_array"

    def create_input_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        inputs = {
            "tokens_array": {"type": "array", "doc": "An array of lists of tokens."}
        }
        return inputs

    def create_output_schema(
        self,
    ) -> typing.Mapping[
        str, typing.Union[ValueSchema, typing.Mapping[str, typing.Any]]
    ]:
        outputs = {
            "tokens_array": {
                "type": "array",
                "doc": "An array of lists of lemmatized tokens.",
            }
        }
        return outputs

    def process(self, inputs: ValueSet, outputs: ValueSet) -> None:

        tokens: pa.Array = inputs.get_value_data("tokens_array")

        # TODO: install this on demand?
        import it_core_news_sm

        it_nlp = it_core_news_sm.load(disable=["tagger", "parser", "ner"])

        class CustomTokenizer(DummyTokenizer):
            def __init__(self, vocab):
                self.vocab = vocab

            def __call__(self, words):
                return Doc(self.vocab, words=words)

        it_nlp.tokenizer = CustomTokenizer(it_nlp.vocab)
        result = []

        for doc in it_nlp.pipe(
            tokens.to_pylist(),
            batch_size=32,
            n_process=3,
            disable=["parser", "ner", "tagger"],
        ):
            result.append([tok.lemma_ for tok in doc])

        outputs.set_value("tokens_array", pa.array(result))
