from jute_disease.models.dl.backbones.jute_classifier import JuteClassifier


# TODO: Finalize once the model structure has been finished.
class InferenceService:
    def __init__(self, checkpoint_path=None):
        self.model = None
        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path):
        self.model = JuteClassifier.load_from_checkpoint(checkpoint_path)
        self.model.eval()

    def predict(self, image_path):
        if self.model is None:
            return "No Model Loaded", 0.0

        # TODO: Add logic to open image, transform, and run forward pass.
        return "Results", 1.0
