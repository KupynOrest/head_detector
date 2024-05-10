import os

import cv2
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


class SDSafetyFilter:
    def __init__(self):
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, image, safety_type: str = "black"):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.safety_checker.device)
        np_images, has_nsfw_concept = self.safety_checker(
            images=[image], clip_input=safety_checker_input.pixel_values.to(torch.float16)
        )
        return has_nsfw_concept[0]


class SafetyClassifier:
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.model = self._load_model("/work/okupyn/mobilenet_v2_140_224")
        self.image_size = 224

    def _load_model(self, model_path):
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("saved_model_path must be the valid directory of a saved model to load.")

        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
        return model

    def classify(self, image):
        """
        Classify given a model, input paths (could be single string), and image dimensionality.

        Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
        """
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = keras.preprocessing.image.img_to_array(image)
        image /= 255
        probs = self.classify_nd(np.asarray([image]))
        return probs

    def classify_nd(self, nd_images):
        """
        Classify given a model, image array (numpy)

        Optionally, pass predict_args that will be passed to tf.keras.Model.predict().
        """
        model_preds = self.model.predict(nd_images)

        categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

        probs = []
        for i, single_preds in enumerate(model_preds):
            single_probs = {}
            for j, pred in enumerate(single_preds):
                single_probs[categories[j]] = float(pred)
            probs.append(single_probs)
        return probs

    def __call__(self, image, safety_type: str = "black"):
        prediction = self.classify(image)
        prediction = max(prediction[0], key=prediction[0].get)
        return not (prediction == "drawings" or prediction == "neutral")


class UnsafeContentDetector:
    def __init__(self):
        self.sd_safety_filter = SDSafetyFilter()
        self.safety_classifier = SafetyClassifier()

    def __call__(self, image: np.ndarray) -> bool:
        return self.sd_safety_filter(image) or self.safety_classifier(image)
