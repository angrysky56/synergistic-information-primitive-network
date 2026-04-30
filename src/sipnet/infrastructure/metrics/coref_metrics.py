from typing import List, Set, Tuple, Dict
import numpy as np

def b3(clusters: List[Set[int]], gold_clusters: List[Set[int]]) -> Tuple[float, float, float]:
    """Calculates B3 coreference metric."""
    if not clusters or not gold_clusters:
        return 0.0, 0.0, 0.0
        
    gold_map = {}
    for i, cluster in enumerate(gold_clusters):
        for mention in cluster:
            gold_map[mention] = cluster
            
    pred_map = {}
    for i, cluster in enumerate(clusters):
        for mention in cluster:
            pred_map[mention] = cluster
            
    all_mentions = set(gold_map.keys()) | set(pred_map.keys())
    
    precision = 0.0
    recall = 0.0
    
    for m in all_mentions:
        if m not in pred_map:
            continue
        if m not in gold_map:
            # Mention in pred but not in gold
            continue
            
        p_cluster = pred_map[m]
        g_cluster = gold_map[m]
        
        intersection = p_cluster & g_cluster
        precision += len(intersection) / len(p_cluster)
        recall += len(intersection) / len(g_cluster)
        
    precision /= len(all_mentions)
    recall /= len(all_mentions)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def muc(clusters: List[Set[int]], gold_clusters: List[Set[int]]) -> Tuple[float, float, float]:
    """Calculates MUC coreference metric."""
    def get_links(cluster_list: List[Set[int]]) -> int:
        links = 0
        for c in cluster_list:
            links += len(c) - 1
        return max(0, links)

    def get_recall(p_clusters: List[Set[int]], g_clusters: List[Set[int]]) -> float:
        num = 0
        for g in g_clusters:
            # Number of partitions of g in p_clusters
            partitions = 0
            for p in p_clusters:
                if g & p:
                    partitions += 1
            num += len(g) - max(1, partitions)
        den = get_links(g_clusters)
        return num / den if den > 0 else 0.0

    recall = get_recall(clusters, gold_clusters)
    precision = get_recall(gold_clusters, clusters)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# Simple wrapper for all metrics
def evaluate_coref(clusters: List[Set[int]], gold_clusters: List[Set[int]]) -> Dict[str, float]:
    p_b3, r_b3, f_b3 = b3(clusters, gold_clusters)
    p_muc, r_muc, f_muc = muc(clusters, gold_clusters)
    
    return {
        "b3_f1": f_b3,
        "muc_f1": f_muc,
        "avg_f1": (f_b3 + f_muc) / 2
    }
