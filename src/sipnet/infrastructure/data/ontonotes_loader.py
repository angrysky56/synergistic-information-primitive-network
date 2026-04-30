import os
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CoNLL2012Dataset(Dataset[Dict[str, Any]]):
    """
    Dataset loader for OntoNotes 5.0 (CoNLL-2012 format).
    Processes documents into mentions and clusters for coreference resolution.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
    ):
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.documents = self._parse_conll(file_path)

    def _parse_conll(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses a CoNLL-2012 file into a list of documents."""
        if not os.path.exists(file_path):
            # For Phase 6.1 sanity check, if file doesn't exist, return empty or mock
            print(f"Warning: {file_path} not found. Returning empty document list.")
            return []

        documents = []
        current_doc: Dict[str, Any] = {
            "id": None,
            "words": [],
            "clusters": {},  # entity_id -> list of (start, end) inclusive
            "open_clusters": {}, # entity_id -> list of start indices
        }

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#begin document"):
                    match = re.search(r"#begin document \((.*)\);", line)
                    current_doc["id"] = match.group(1) if match else "unknown"
                    continue
                
                if line.startswith("#end document"):
                    # Close any remaining open clusters (shouldn't happen in valid CoNLL)
                    documents.append({
                        "id": current_doc["id"],
                        "words": current_doc["words"],
                        "clusters": current_doc["clusters"]
                    })
                    current_doc = {"id": None, "words": [], "clusters": {}, "open_clusters": {}}
                    continue

                if not line:
                    continue

                parts = line.split()
                if len(parts) < 12:
                    continue

                word = parts[3]
                coref = parts[-1]
                word_idx = len(current_doc["words"])
                current_doc["words"].append(word)

                if coref != "-":
                    # Parse coreference info e.g. (7)|(8), (7), 7), (7)
                    tags = coref.split("|")
                    for tag in tags:
                        if tag.startswith("(") and tag.endswith(")"):
                            entity_id = int(tag[1:-1])
                            if entity_id not in current_doc["clusters"]:
                                current_doc["clusters"][entity_id] = []
                            current_doc["clusters"][entity_id].append((word_idx, word_idx))
                        elif tag.startswith("("):
                            entity_id = int(tag[1:])
                            if entity_id not in current_doc["open_clusters"]:
                                current_doc["open_clusters"][entity_id] = []
                            current_doc["open_clusters"][entity_id].append(word_idx)
                        elif tag.endswith(")"):
                            entity_id = int(tag[:-1])
                            if entity_id in current_doc["open_clusters"] and current_doc["open_clusters"][entity_id]:
                                start_idx = current_doc["open_clusters"][entity_id].pop()
                                if entity_id not in current_doc["clusters"]:
                                    current_doc["clusters"][entity_id] = []
                                current_doc["clusters"][entity_id].append((start_idx, word_idx))

        return documents

    def _align_subtokens(self, words: List[str]) -> Tuple[torch.Tensor, List[int]]:
        """Aligns subtokens to original word indices."""
        tokens = []
        word_to_subtoken_map = []
        
        # Add [CLS]
        tokens.append(self.tokenizer.cls_token_id)
        
        for i, word in enumerate(words):
            word_subtokens = self.tokenizer.encode(word, add_special_tokens=False)
            word_to_subtoken_map.append(len(tokens)) # Map word index to first subtoken index
            tokens.extend(word_subtokens)
            
        # Add [SEP]
        tokens.append(self.tokenizer.sep_token_id)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.tokenizer.sep_token_id]
            # Filter map to only include tokens within max_length
            word_to_subtoken_map = [idx for idx in word_to_subtoken_map if idx < self.max_length - 1]

        return torch.tensor(tokens, dtype=torch.long), word_to_subtoken_map

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.documents[idx]
        input_ids, word_map = self._align_subtokens(doc["words"])
        
        return {
            "doc_id": doc["id"],
            "input_ids": input_ids,
            "word_map": word_map,
            "clusters": doc["clusters"]
        }

def ontonotes_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for CoNLL2012Dataset."""
    input_ids = [item["input_ids"] for item in batch]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    
    # We can't easily pad word_map and clusters due to their varying structure
    # They will be returned as lists
    return {
        "input_ids": padded_input_ids,
        "word_maps": [item["word_map"] for item in batch],
        "clusters": [item["clusters"] for item in batch],
        "doc_ids": [item["doc_id"] for item in batch]
    }
