import torch
from torch.utils.data import DataLoader
from data import RetinaDataset
from sampler import WeightedRandomSampler, LPRandomOverSampler
from transform import Transform

def create_dataloader(data_dir, batch_size, num_workers, size, phase):
    transform = Transform(size=size, phase=phase)

    if phase == 'train':
        dataset = RetinaDataset(data_dir=data_dir, split=phase, transform=transform, known_labels = 100, testing =False)
        sampler = WeightedRandomSampler(labels=dataset.labels)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        dataset = RetinaDataset(data_dir=data_dir, split=phase, transform=transform, known_labels = 0, testing =True)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader



def create_dataloader2(data_dir, batch_size, num_workers, size, phase):
    transform = Transform(size=size, phase=phase)

    dataset = RetinaDataset(data_dir=data_dir, split=phase, transform=transform)

    if phase == 'train':
        sampler = LPRandomOverSampler(labels=dataset.labels, sample_percent=0.1)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader