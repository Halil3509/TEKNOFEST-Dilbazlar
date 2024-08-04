from utils import get_configs
import logging
import os
from dotenv import load_dotenv
load_dotenv()


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SOTA:
    def __init__(self):
        self.anxiety_labels = ['Agorafobi', 'Panik', 'Fobi', 'Seçici Dilsizlik', 'Sosyal Anksiyete']
        self.depression_labels = ['Distimi', 'PMDD']
        self.disorder_or_not_labels = ["Normal", "Hastalık"]
        self.anx_dep_labels = ["Depresyon", "Anksiyete"]

        self._setup_logger()
        self._connect_hugging_face()
        self.configs = get_configs()
        (self.anxiety_trainer, self.depression_trainer,
         self.disorder_or_not_trainer, self.anx_dep_trainer) = self.load_models()

    def _setup_logger(self):
        # Configure the logger
        self.logger = logging.getLogger('HuggingFaceLogin')
        self.logger.setLevel(logging.DEBUG)

        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        self.logger.addHandler(ch)

    @staticmethod
    def _connect_hugging_face():
        api = HfApi(os.environ.get('HG_ACCESS_TOKEN'))
        print("Successfully logged in Huggingface!")

    def load_models(self):
        anxiety_model = self._load_single_model(model_url=self.configs["ANXIETY_MODEL_ID"], num_labels=5)
        depression_model = self._load_single_model(model_url=self.configs["DEPRESSION_MODEL_ID"], num_labels=2)
        disorder_or_not_model = self._load_single_model(model_url=self.configs["DISORDER_MODEL_ID"], num_labels=2)
        anx_dep_model = self._load_single_model(model_url=self.configs["ANX_DEP_MODEL_ID"], num_labels=2)

        self.logger.info("All huggingface models have been loaded.")
        return anxiety_model, depression_model, disorder_or_not_model, anx_dep_model

    @staticmethod
    def _load_single_model(model_url: str, num_labels: int):

        model = AutoModelForSequenceClassification.from_pretrained(
            model_url,
            num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_url)

        # Move the model to the appropriate device
        global device
        model.to(device)

        # Ensure model is in evaluation mode
        model.eval()

        return {
            'model': model,
            'tokenizer': tokenizer
        }

    @staticmethod
    def predict(model, tokenizer, labels, input_text):
        """

        :param model:
        :param tokenizer:
        :param labels:
        :param input_text:
        :return:
        """
        inputs = tokenizer(input_text, max_length=150, padding="max_length", truncation=True, return_tensors="pt")

        # Move the inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Disable gradient computation for inference
        with torch.no_grad():
            # Forward pass to get outputs
            outputs = model(**inputs)

            # Get the prediction
            # Note: `AutoModel` might not include logits. Ensure you use the appropriate model class for your task.
            if hasattr(outputs, 'logits'):
                preds = torch.argmax(outputs.logits, dim=-1)
            else:
                # Handle the case where the model does not have logits (e.g., outputs are raw hidden states)
                preds = torch.argmax(outputs[0], dim=-1)

        # Convert prediction to numpy array and print (if needed)
        prediction = preds.cpu().numpy()[0]

        return labels[prediction], outputs

    def anxiety_predict(self, sentence):
        """

        :return:
        """

        label, outputs_probs = self.predict(model=self.anxiety_trainer['model'],
                                            tokenizer=self.anxiety_trainer['tokenizer'],
                                            labels=self.anxiety_labels,
                                            input_text=sentence)

        return label

    def depression_predict(self, sentence):
        """

        :param sentence:
        :return:
        """
        label, outputs_probs = self.predict(model=self.depression_trainer['model'],
                                            tokenizer=self.depression_trainer['tokenizer'],
                                            labels=self.depression_labels,
                                            input_text=sentence)

        return label

    def anx_dep_predict(self, sentence):
        """

        :param sentence:
        :return:
        """
        label, outputs_probs = self.predict(model=self.anx_dep_trainer['model'],
                                            tokenizer=self.anx_dep_trainer['tokenizer'],
                                            labels=self.anx_dep_labels,
                                            input_text=sentence)

        return label

    def disorder_or_not_predict(self, sentence):
        """

        :param sentence:
        :return:
        """
        label, outputs_probs = self.predict(model=self.disorder_or_not_trainer['model'],
                                            tokenizer=self.disorder_or_not_trainer['tokenizer'],
                                            labels=self.disorder_or_not_labels,
                                            input_text=sentence)

        return label

    def prediction_flow_standard(self, sentence):
        """

        :param sentence:
        :return:
        """

        # Is it disorder or not
        is_disorder_result_label = self.disorder_or_not_predict(sentence)

        if is_disorder_result_label == 'Normal':
            self.logger.info(f"This sentence is {is_disorder_result_label}")

        else: # Hastalık
            self.logger.info(f"This sentence is {is_disorder_result_label}")

            dep_or_anx_result_label = self.anx_dep_predict(sentence)
            self.logger.info(f"This sentence is {dep_or_anx_result_label}")

            if dep_or_anx_result_label == 'Depresyon':
                depression_result_label = self.depression_predict(sentence)
                self.logger.info(f"This sentence is {depression_result_label}")

            else: # Anksiyete
                anxiety_result_label = self.anxiety_predict(sentence)
                self.logger.info(f"This sentence is {anxiety_result_label}")
