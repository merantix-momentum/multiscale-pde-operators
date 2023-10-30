import numpy as np
from tqdm import tqdm

from multiscale_operator.data import get_catalog
from multiscale_operator.data.datasets import MSOIterableDataset


def get_dataset_stats(dataset):
    """Iterate through dataset and compute mean and std."""
    data = []
    for sample in tqdm(dataset):
        data += sample.x.numpy().flatten().tolist()
    return np.mean(data), np.std(data)


if __name__ == "__main__":
    catalog = get_catalog()
    driver = catalog["MotorDataset"].get_driver()
    dataset = MSOIterableDataset(driver, split="train")
    mean, std = get_dataset_stats(dataset)
    print("Dataset mean: ", mean)
    print("Dataset std: ", std)
