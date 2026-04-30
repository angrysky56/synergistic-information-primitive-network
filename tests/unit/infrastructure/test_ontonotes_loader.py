import pytest
import torch
import os
from sipnet.infrastructure.data.ontonotes_loader import CoNLL2012Dataset, ontonotes_collate_fn

def create_dummy_conll(path: str):
    content = """#begin document (test_doc);
test_doc\t0\t0\tThe\tDT\t(TOP(S(NP*)\t-\t-\t-\tSpeaker\t*\t-\t(7
test_doc\t0\t1\tquick\tJJ\t*\t-\t-\t-\tSpeaker\t*\t-\t-
test_doc\t0\t2\tbrown\tJJ\t*\t-\t-\t-\tSpeaker\t*\t-\t-
test_doc\t0\t3\tfox\tNN\t*))\t-\t-\t-\tSpeaker\t*\t-\t7)
test_doc\t0\t4\tjumps\tVBZ\t(VP*)\t-\t-\t-\tSpeaker\t*\t-\t-
test_doc\t0\t5\tover\tIN\t(PP*)\t-\t-\t-\tSpeaker\t*\t-\t-
test_doc\t0\t6\tthe\tDT\t(NP*\t-\t-\t-\tSpeaker\t*\t-\t(8
test_doc\t0\t7\tlazy\tJJ\t*\t-\t-\t-\tSpeaker\t*\t-\t-
test_doc\t0\t8\tdog\tNN\t*)\t-\t-\t-\tSpeaker\t*\t-\t8)
test_doc\t0\t9\t.\t.\t*\t-\t-\t-\tSpeaker\t*\t-\t-

#end document
"""
    with open(path, "w") as f:
        f.write(content)

def test_ontonotes_loader(tmp_path):
    conll_file = tmp_path / "test.conll"
    create_dummy_conll(str(conll_file))
    
    dataset = CoNLL2012Dataset(str(conll_file), tokenizer_name="bert-base-uncased")
    assert len(dataset) == 1
    
    item = dataset[0]
    assert item["doc_id"] == "test_doc"
    assert "input_ids" in item
    assert "word_map" in item
    assert 7 in item["clusters"]
    assert item["clusters"][7] == [(0, 3)]
    assert 8 in item["clusters"]
    assert item["clusters"][8] == [(6, 8)]

def test_ontonotes_collate():
    batch = [
        {
            "doc_id": "doc1",
            "input_ids": torch.tensor([1, 2, 3]),
            "word_map": [0, 1, 2],
            "clusters": {1: [(0, 1)]}
        },
        {
            "doc_id": "doc2",
            "input_ids": torch.tensor([4, 5]),
            "word_map": [0, 1],
            "clusters": {2: [(1, 1)]}
        }
    ]
    
    collated = ontonotes_collate_fn(batch)
    assert collated["input_ids"].shape == (2, 3)
    assert len(collated["word_maps"]) == 2
    assert len(collated["clusters"]) == 2
    assert collated["doc_ids"] == ["doc1", "doc2"]
