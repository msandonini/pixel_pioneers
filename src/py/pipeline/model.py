"""
from typing import Callable, Literal

import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image


class PipelineVisionEncoderConfiguration():
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Callable
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        

class PipelineVisionEncoder(nn.Module):
    def __init__(
        self, 
        model_config: PipelineVisionEncoderConfiguration,
        # transform=None, 
        device: Literal['cuda', 'cpu'] | None = None
    ):
        super().__init__()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)

        self.model = model_config.model.to(self.device).eval()
        self.features = {}

        self.feature_extractor = model_config.feature_extractor
        # hook_layer.register_forward_hook(
        #     self._get_activation("inter_feat")
        # )

        # if transform is None:
        #     # Set standard transforms
        #     self.transform = transform if transform is not None else transforms.Compose([
        #         transforms.Resize( (224, 224) ),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean = [ 0.485, 0.456, 0.406 ],
        #             std = [ 0.229, 0.224, 0.225 ]
        #         )
        #     ])
        # else:
        #     self.transform = transform
    

    # def _get_activation(self, name):
    #     def hook(model, input, output):
    #         self.features[name] = output
    #     return hook

    # def forward(self, image, extract_intermediate: bool = True):
    def forward(self, image):
        img = Image.open(image).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # output = self.model(tensor)
            return self.feature_extractor(self.model, tensor)

        # if extract_intermediate:
        #     # Intermediate layer
        #     return self.features['inter_feat']
        
        # # Final layer
        # return output


if __name__ == "__main__":

    # SigLIP 2
    
    import timm

    models = {
        "siglip2": PipelineVisionEncoderConfiguration(
            timm.create_model(
                "vit_base_patch16_siglip_224",
                pretrained = True
            ),
            feature_extractor = lambda m, x: m.forward_features(x)
        ),
        "dinov2": PipelineVisionEncoderConfiguration(
            torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14"
            ),
            feature_extractor = lambda m, x: m.get_intermediate_layers(x, n = 1)[0]
        ),
        "meta-perc": PipelineVisionEncoderConfiguration(
            timm.create_model(
                "vit_base_patch16_224", 
                pretrained = True
            ),
            feature_extractor=lambda m, x: m.forward_features(x)
        ),
        "tipsv2": PipelineVisionEncoderConfiguration(
            timm.create_model(
                "vit_large_patch16_224",
                pretrained = True
            ),
            feature_extractor=lambda m, x: m.forward_features(x)
        )
    }

    encoders = {}
    for name, modconf in models.items():
        encoders[name] = PipelineVisionEncoder(
            model = modconf.model,
            feature_extractor = modconf.feature_extractor
        )
"""

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