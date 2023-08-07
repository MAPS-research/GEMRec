
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = F.normalize(image_embeds)
    normalized_text_embeds = F.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

class NSFWScore(nn.Module):
    def __init__(self, checker, adjustment=0.0):
        super().__init__()
        self.checker = checker
        self.adjustment = adjustment

    @torch.no_grad()
    def forward(self, clip_input: torch.FloatTensor):
        pooled_output = self.checker.vision_model(clip_input)['pooler_output']
        image_embeds = self.checker.visual_projection(pooled_output)
        special_cos_dist = cosine_distance(image_embeds, self.checker.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.checker.concept_embeds)
        special_scores = special_cos_dist - self.checker.special_care_embeds_weights + self.adjustment
        special_adjustment = torch.any(special_scores > 0, dim=1) * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])
        concept_scores = (cos_dist - self.checker.concept_embeds_weights) + special_adjustment
        nsfw_scores = (concept_scores.max(dim=1).values + 1) / 2    # [-1, 1] => [0, 1]
        return nsfw_scores
