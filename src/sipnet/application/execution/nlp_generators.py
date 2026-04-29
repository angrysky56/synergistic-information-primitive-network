import random

import torch
from torch.utils.data import Dataset


class SequentialNLPDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Synthetic NLP Coreference Generator.
    Generates sequences like: ["The", "cat", "saw", "the", "dog", "until", "it", "slept"]
    Target: At the timestep of the pronoun ("it"), the network must output
    the class ID of the primary subject.
    """

    def __init__(self, num_samples: int, include_distractors: bool = True):
        self.num_samples = num_samples
        self.include_distractors = include_distractors

        # Define a synthetic vocabulary
        self.articles = ["The", "A", "the", "a"]
        self.subjects = ["dog", "cat", "robot", "agent", "bird", "human"]
        self.verbs1 = [
            "barked",
            "purred",
            "processed",
            "computed",
            "watched",
            "followed",
        ]
        self.conjunctions = ["until", "before", "while", "and", "because"]
        self.pronouns = ["it", "he", "she"]
        self.verbs2 = ["slept", "halted", "crashed", "rested", "ran", "jumped"]

        # Build vocabulary mapping
        self.vocab = (
            ["<PAD>"]
            + self.articles
            + self.subjects
            + self.verbs1
            + self.conjunctions
            + self.pronouns
            + self.verbs2
        )
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # Generate data
        self.data, self.targets = self._generate_dataset()

    def _generate_dataset(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        data_seqs = []
        target_seqs = []

        for _ in range(self.num_samples):
            # Construct a synthetic sentence with potential distractor
            primary_subj = random.choice(self.subjects)

            if self.include_distractors and random.random() > 0.5:
                # Sentence with distractor: "The [subj1] [verb1] the [subj2] [conj] it [verb2]"
                distractor_subj = random.choice(
                    [s for s in self.subjects if s != primary_subj]
                )
                sentence = [
                    random.choice(self.articles).capitalize(),
                    primary_subj,
                    random.choice(self.verbs1),
                    random.choice(self.articles),
                    distractor_subj,
                    random.choice(self.conjunctions),
                    random.choice(self.pronouns),
                    random.choice(self.verbs2),
                ]
                pronoun_idx = 6
            else:
                # Standard sentence: "The [subj] [verb1] [conj] it [verb2]"
                sentence = [
                    random.choice(self.articles).capitalize(),
                    primary_subj,
                    random.choice(self.verbs1),
                    random.choice(self.conjunctions),
                    random.choice(self.pronouns),
                    random.choice(self.verbs2),
                ]
                pronoun_idx = 4

            # Convert to indices
            indices = torch.tensor(
                [self.word2idx[word] for word in sentence], dtype=torch.long
            )
            data_seqs.append(indices)

            # Create targets: all zeros except at the pronoun where target is primary_subj
            target = torch.zeros(len(sentence), dtype=torch.long)
            target[pronoun_idx] = self.word2idx[primary_subj]
            target_seqs.append(target)

        return data_seqs, target_seqs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def nlp_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad variable-length sequences.
    """
    data, targets = zip(*batch)

    # Pad sequences to the max length in the batch
    padded_data = torch.nn.utils.rnn.pad_sequence(
        list(data), batch_first=True, padding_value=0
    )
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        list(targets), batch_first=True, padding_value=0
    )

    return padded_data, padded_targets
