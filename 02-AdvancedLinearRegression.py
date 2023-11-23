import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

cost_lst = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    Hypothesis = model(x_train)
    cost = F.mse_loss(Hypothesis, y_train)
    cost_lst.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


plt.title("cost")
plt.plot(cost_lst)
plt.show()
