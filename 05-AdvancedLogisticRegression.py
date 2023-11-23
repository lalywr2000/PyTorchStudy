import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)

cost_lst = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    Hypothesis = model(x_train)

    cost = F.binary_cross_entropy(Hypothesis, y_train)
    cost_lst.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


# prediction = Hypothesis >= torch.FloatTensor([0.5])
# correct_prediction = prediction.float() == y_train
# accuracy = correct_prediction.mean()


plt.title("cost")
plt.plot(cost_lst)
plt.show()
