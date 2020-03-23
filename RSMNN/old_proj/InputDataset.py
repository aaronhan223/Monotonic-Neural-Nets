from torch.utils.data import Dataset


class InputDataset(Dataset):
    """Helper class for input datasets"""

    def __init__(self, inputs, results):

        self.inputs = inputs
        self.results = results

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        return {'InputVector': self.inputs[idx], 'Result': self.results[idx]}
