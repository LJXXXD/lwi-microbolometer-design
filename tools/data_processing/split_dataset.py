import torch


def split_dataset(dataset, train_perc, random_flag=False, random_seed=0):
    train_size = int(train_perc * len(dataset))
    test_size = len(dataset) - train_size
    if random_flag:
        torch.manual_seed(random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset