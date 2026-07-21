from typing import Callable, Literal
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class PipelineVisionEncoderConfiguration():
    def __init__(self, model: nn.Module, feature_extractor: Callable):
        self.model = model
        self.feature_extractor = feature_extractor
        
class PipelineVisionEncoder(nn.Module):
    def __init__(self, model_config: PipelineVisionEncoderConfiguration, device: Literal['cuda', 'cpu'] | None = None):
        super().__init__()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)

        self.model = model_config.model.to(self.device).eval()
        self.feature_extractor = model_config.feature_extractor

        # REGOLA DELLE TRASFORMAZIONI STANDARD (Ripristinata perché indispensabile)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, image_or_tensor):
        # Se la pipeline passa un path/stringa (immagine singola)
        if isinstance(image_or_tensor, (str, Image.Image)) or hasattr(image_or_tensor, 'open'):
            img = Image.open(image_or_tensor).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
        else:
            # Se la pipeline passa già un tensore dal DataLoader (il tuo caso nel main!)
            tensor = image_or_tensor.to(self.device)
        
        with torch.no_grad():
            return self.feature_extractor(self.model, tensor)