import torch
from PIL import Image

from jute_disease.data import dl_val_transforms
from jute_disease.models.dl import Classifier


class InferenceService:
    def __init__(self, checkpoint_path=None, class_names=None):
        self.model = None
        self.class_names = class_names
        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        self.model = Classifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path):
        if self.model is None:
            return "No Model Loaded", 0.0

        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = dl_val_transforms(img).unsqueeze(0)  # Add batch dim

        # Forward pass
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        pred_idx = pred.item()
        confidence = conf.item()

        # Map to class name if available
        if self.class_names:
            label = self.class_names[pred_idx]
        else:
            label = str(pred_idx)

        return label, confidence
