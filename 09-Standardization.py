import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

training_loss = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    prediction = model(norm_x_train)

    cost = F.mse_loss(prediction, y_train)
    training_loss.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


plt.title("loss")
plt.plot(training_loss)
plt.show()
