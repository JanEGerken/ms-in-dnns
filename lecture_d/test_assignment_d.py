import income_net


def test_resampled_dataset():
    import torch
    from torch.utils.data import TensorDataset

    test_dataset = TensorDataset(torch.arange(11), torch.tensor([[1, 0]] * 9 + [[0, 1]] * 2))
    resampled_dataset = income_net.ResampledDataset(test_dataset)
    assert len(resampled_dataset) == 18
    labels = torch.argmax(torch.stack([sample[1] for sample in resampled_dataset]), dim=-1)
    n_class_0 = (labels == 0).sum()
    n_class_1 = (labels == 1).sum()
    assert n_class_0 == n_class_1
    x = torch.stack([sample[0] for sample in resampled_dataset])
    unique_x, counts_x = torch.unique(x, return_counts=True)
    assert torch.all(unique_x == torch.arange(11))
    assert torch.all(counts_x == torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 4])) or torch.all(
        counts_x == torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 5])
    )


if __name__ == "__main__":
    test_resampled_dataset()
