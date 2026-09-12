import torch
import torch.nn as nn

class KureselGeometrikLoss(nn.Module):
    def __init__(self):
        super(KureselGeometrikLoss, self).__init__()
        # Doğrudan optimize edilmiş saf CrossEntropy
        self.temel_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, model_ciktisi, gercek_hedef):
        return self.temel_loss(model_ciktisi, gercek_hedef)