import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, Blip2ForConditionalGeneration


CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',    # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',               # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',       # 15.77GB
    'git-large-coco': 'microsoft/git-large-coco',            # 1.58GB
    'fuse-cap': 'noamrot/FuseCap'                            # 990MB
}


class ImageCaptioner:
    def __init__(self, model: str = "blip2-2.7b", device: str = "cuda"):
        self.model = model
        self.device = device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_path = CAPTION_MODELS[model]
        if model.startswith('git-'):
            caption_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        elif model.startswith('blip2-'):
            caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        else:
            caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        self.caption_processor = AutoProcessor.from_pretrained(model_path)
        self.caption_model = caption_model.eval().to(device)

    def generate_caption(self, image: np.ndarray) -> str:
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        if not self.model.startswith('git-'):
            inputs = inputs.to(self.dtype)
        tokens = self.caption_model.generate(**inputs, max_new_tokens=50)
        return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
