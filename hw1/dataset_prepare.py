import pandas as pd

import loader
import scaling


def purge_dataset(dataset: pd.DataFrame):
    for i in dataset.columns:
        if (dataset[i] == 0).all():
            dataset.drop(i, inplace=True, axis=1)


# Remove zero columns from dataset
dataset5 = (
    loader.load_training_dataset("dataset1.csv")
    .append(loader.load_training_dataset("dataset2.csv"))
    .append(loader.load_training_dataset("dataset3.csv"))
    .append(loader.load_training_dataset("dataset4.csv"))
)
purge_dataset(dataset5)
test_dataset5 = loader.load_training_dataset("dataset5.csv")
purge_dataset(test_dataset5)

dataset4 = (
    loader.load_training_dataset("dataset1.csv")
    .append(loader.load_training_dataset("dataset2.csv"))
    .append(loader.load_training_dataset("dataset3.csv"))
    .append(loader.load_training_dataset("dataset5.csv"))
)
purge_dataset(dataset4)
test_dataset4 = loader.load_training_dataset("dataset4.csv")
purge_dataset(test_dataset4)

dataset3 = (
    loader.load_training_dataset("dataset1.csv")
    .append(loader.load_training_dataset("dataset2.csv"))
    .append(loader.load_training_dataset("dataset4.csv"))
    .append(loader.load_training_dataset("dataset5.csv"))
)
purge_dataset(dataset3)
test_dataset3 = loader.load_training_dataset("dataset3.csv")
purge_dataset(test_dataset3)

dataset2 = (
    loader.load_training_dataset("dataset1.csv")
    .append(loader.load_training_dataset("dataset3.csv"))
    .append(loader.load_training_dataset("dataset4.csv"))
    .append(loader.load_training_dataset("dataset5.csv"))
)
purge_dataset(dataset2)
test_dataset2 = loader.load_training_dataset("dataset2.csv")
purge_dataset(test_dataset2)

dataset1 = (
    loader.load_training_dataset("dataset3.csv")
    .append(loader.load_training_dataset("dataset2.csv"))
    .append(loader.load_training_dataset("dataset4.csv"))
    .append(loader.load_training_dataset("dataset5.csv"))
)
purge_dataset(dataset1)
test_dataset1 = loader.load_training_dataset("dataset1.csv")
purge_dataset(test_dataset1)

# Normalize dataset
dataset1 = scaling.normalize(dataset1)
test_dataset1 = scaling.normalize(test_dataset1)

dataset2 = scaling.normalize(dataset2)
test_dataset2 = scaling.normalize(test_dataset2)

dataset3 = scaling.normalize(dataset3)
test_dataset3 = scaling.normalize(test_dataset3)

dataset4 = scaling.normalize(dataset4)
test_dataset4 = scaling.normalize(test_dataset4)

dataset5 = scaling.normalize(dataset5)
test_dataset5 = scaling.normalize(test_dataset5)

# Save dataset to file
loader.save_normalized_dataset(dataset1, "dataset1.csv")
loader.save_normalized_dataset(test_dataset1, "test_dataset1.csv")

loader.save_normalized_dataset(dataset2, "dataset2.csv")
loader.save_normalized_dataset(test_dataset2, "test_dataset2.csv")

loader.save_normalized_dataset(dataset3, "dataset3.csv")
loader.save_normalized_dataset(test_dataset3, "test_dataset3.csv")

loader.save_normalized_dataset(dataset4, "dataset4.csv")
loader.save_normalized_dataset(test_dataset4, "test_dataset4.csv")

loader.save_normalized_dataset(dataset5, "dataset5.csv")
loader.save_normalized_dataset(test_dataset5, "test_dataset5.csv")
