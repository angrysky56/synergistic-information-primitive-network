import torch
from torch.utils.data import Dataset
import random

class SequentialNLPDataset(Dataset):
    """
    Synthetic NLP Coreference Generator.
    Generates sequences like: ["The", "dog", "barked", "until", "it", "slept"]
    Target: At the timestep of the pronoun ("it"), the network must output
    the class ID of the original subject ("dog").
    """
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

        # Define a simple synthetic vocabulary
        self.articles = ["The", "A"]
        self.subjects = ["dog", "cat", "robot", "agent"]
        self.verbs1 = ["barked", "purred", "processed", "computed"]
        self.conjunctions = ["until", "before", "while", "and"]
        self.pronouns = ["it", "he", "she"]
        self.verbs2 = ["slept", "halted", "crashed", "rested"]

        # Build vocabulary mapping
        self.vocab = ["<PAD>"] + self.articles + self.subjects + self.verbs1 + \
                     self.conjunctions + self.pronouns + self.verbs2
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.seq_len = 6 # Fixed sequence length for this synthetic task

        # Generate data
        self.data, self.targets = self._generate_dataset()

    def _generate_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        data_seqs = []
        target_seqs = []

        for _ in range(self.num_samples):
            # Construct a synthetic sentence
            subj = random.choice(self.subjects)
            sentence = [
                random.choice(self.articles),
                subj,
                random.choice(self.verbs1),
                random.choice(self.conjunctions),
                random.choice(self.pronouns),
                random.choice(self.verbs2)
            ]

            # Convert to indices
            indices = [self.word2idx[word] for word in sentence]
            data_seqs.append(indices)

            # Create targets: all zeros (padding/ignore) except at the pronoun (index 4)
            # where the target is the ID of the subject.
            target = [0] * self.seq_len
            target[4] = self.word2idx[subj]
            target_seqs.append(target)

        return torch.tensor(data_seqs, dtype=torch.long), torch.tensor(target_seqs, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]
