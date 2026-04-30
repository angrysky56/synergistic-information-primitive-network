import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class CorefMentionRankingHead(nn.Module):
    """
    Mention-ranking head for coreference resolution.
    Scores pairs of mentions (antecedent, mention).
    """

    def __init__(self, hidden_dim: int, feature_dim: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Scoring network: concat(m_i, m_j, m_i * m_j, features)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3 + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Metadata embeddings (e.g., distance, speaker)
        self.distance_embeddings = nn.Embedding(10, feature_dim)

    def forward(
        self,
        mention_embeddings: torch.Tensor,
        mention_indices: List[int],
        max_antecedents: int = 50
    ) -> torch.Tensor:
        """
        Calculates coreference scores for mentions.
        
        Args:
            mention_embeddings: (num_mentions, hidden_dim)
            mention_indices: list of word indices for each mention
            max_antecedents: maximum number of previous mentions to consider
            
        Returns:
            scores: (num_mentions, max_antecedents + 1)
                   Index 0 is the "dummy" antecedent (new cluster).
        """
        num_mentions = mention_embeddings.size(0)
        device = mention_embeddings.device
        
        # scores[i, j] is score of mention i having antecedent j
        # j=0 is dummy
        scores = torch.zeros(num_mentions, max_antecedents + 1, device=device)
        
        for i in range(num_mentions):
            # Candidates for mention i are mentions 0 to i-1
            num_candidates = min(i, max_antecedents)
            if num_candidates == 0:
                continue
                
            curr_emb = mention_embeddings[i].unsqueeze(0).repeat(num_candidates, 1) # (C, H)
            ant_indices = list(range(max(0, i - max_antecedents), i))
            ant_embs = mention_embeddings[ant_indices] # (C, H)
            
            # Simple features: distance bucket
            distances = torch.tensor([i - j for j in ant_indices], device=device)
            dist_buckets = torch.clamp(distances, 0, 9)
            dist_feats = self.distance_embeddings(dist_buckets) # (C, F)
            
            # Combined representation
            combined = torch.cat([
                curr_emb,
                ant_embs,
                curr_emb * ant_embs,
                dist_feats
            ], dim=-1) # (C, 3H + F)
            
            pair_scores = self.scorer(combined).squeeze(-1) # (C)
            scores[i, 1:num_candidates+1] = pair_scores
            
        return scores
