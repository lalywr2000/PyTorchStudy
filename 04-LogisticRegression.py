import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

cost_lst = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    # Hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
    Hypothesis = torch.sigmoid(x_train.matmul(W) + b)

    # losses = -(y_train * torch.log(Hypothesis) +
    #          (1 - y_train) * torch.log(1 - Hypothesis))
    # cost = losses.mean()

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
