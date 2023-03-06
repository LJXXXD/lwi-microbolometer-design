import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, samples, labels):
        
        super(Dataset, self).__init__()

        self.dataset = []
        
        for sample, label in zip(samples, labels):
            self.dataset.append((torch.tensor(sample).float(), torch.tensor(label).float()))

    def __getitem__(self, index):
                                        
        return self.dataset[index]
        
    def __len__(self):
        
        return len(self.dataset)