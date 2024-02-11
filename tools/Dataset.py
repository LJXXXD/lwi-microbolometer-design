import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, samples, labels):
        
        super(Dataset, self).__init__()

        self.dataset = []
        # self.normalized_dataset = []
        
        for sample, label in zip(samples, labels):
            self.dataset.append((torch.tensor(sample).float(), torch.tensor(label).float()))
            # normalized_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
            # self.normalized_dataset.append((torch.tensor(normalized_sample).float(), torch.tensor(label).float()))

    def __getitem__(self, index):
                                        
        return self.dataset[index]
        # return self.normalized_dataset[index]
        
    def __len__(self):
        
        return len(self.dataset)
        # return len(self.normalized_dataset)
    



class NoisyDataset(Dataset):
    def __init__(self, clean_dataset, max_noise):

        super(Dataset, self).__init__()

        self.clean_dataset = clean_dataset
        self.max_noise = max_noise
        self.dataset = []
               
        for sample in self.clean_dataset:
            x = sample[0]
            noisy_x = self.add_noise(x, self.max_noise)

            noisy_sample = (noisy_x.float().to(torch.float32), sample[1])

            self.dataset.append(noisy_sample)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def add_noise(self, x, max_noise):
        if max_noise == 0:
            return x
        else:
            noise = np.random.uniform(-max_noise, max_noise, size=x.shape)
            noisy_x = x + noise
            return noisy_x






