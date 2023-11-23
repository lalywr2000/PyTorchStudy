import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 80],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataLoader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

cost_lst = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    for batch_idx, samples in enumerate(dataLoader):
        x_train, y_train = samples

        Hypothesis = model(x_train)
        cost = F.mse_loss(Hypothesis, y_train)
        cost_lst.append(cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


plt.title("cost")
plt.plot(cost_lst)
plt.show()
