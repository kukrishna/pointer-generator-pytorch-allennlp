from allennlp.predictors.predictor import  Predictor
from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from overrides import overrides
from allennlp.common.util import JsonDict

from sumeval.metrics.rouge import RougeCalculator



@Predictor.register("beamsearch")
class BeamSearchPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.rouge = RougeCalculator(stopwords=True, lang="en")

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"article_lines": "...", "summary_lines": "..."}``.
        """
        return self._dataset_reader.dict_to_instance(json_dict)
    
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        predicted = self.predict_instance(instance)
        ground_truth = " ".join(instance.fields["meta"]["target_tokens"])

        return {"ground_truth": ground_truth,
                "prediction": predicted}
    

