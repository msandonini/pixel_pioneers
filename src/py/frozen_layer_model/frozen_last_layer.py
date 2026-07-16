import torch
import torch.nn as nn
import timm
from pipeline.model import PipelineVisionEncoderConfiguration

def get_frozen_siglip2() -> PipelineVisionEncoderConfiguration:
    """
    Crea un modello SigLIP 2 in cui la testa finale viene congelata (requires_grad = False),
    mentre il resto dei blocchi intermedi rimane libero di aggiornarsi.
    """

    model = timm.create_model(
        "vit_base_patch16_siglip_224", 
        pretrained=True, 
        num_classes=1
    )
    
    print("❄️ Congelamento dell'ultimo layer (head)...")
    for param in model.head.parameters():
        param.requires_grad = False
        
    print("✅ Verifica - Pesi della Head sono addestrabili?", model.head.weight.requires_grad)

    feature_extractor = lambda m, x: m.forward_features(x) # skips the frozen head

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n --- STATISTICHE MODELLO CONGELATO ---")
    print(f" Parametri Totali:       {total_params:,}")
    print(f" Parametri Addestrabili: {trainable_params:,} (I layer intermedi)")
    print(f" Parametri Congelati:    {frozen_params:,} (La testa finale)")
    
    is_head_frozen = not any(p.requires_grad for p in model.head.parameters())
    print(f" Verifica stato 'head':  {'CONGELATA CORRETTAMENTE' if is_head_frozen else 'ERRORE ❌'}")
    print("------------------------------------------\n")
    # ------------------------------------------------


    return PipelineVisionEncoderConfiguration(
        model=model,
        feature_extractor=feature_extractor
    )

if __name__ == "__main__":
    # Test rapido per vedere se funziona
    config_congelata = get_frozen_siglip2()
    print("Configurazione congelata creata con successo per l'esperimento IQA!")