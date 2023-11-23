import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [2, 2, 2, 1, 1, 1, 0, 0]  # one-hot vector: index of 1

x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)


W = torch.zeros((4, 3), requires_grad=True)  # dim, classes
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)

cost_lst = []


nb_epoch = 1000
for epoch in range(1, nb_epoch + 1):
    z = x_train.matmul(W) + b

    cost = F.cross_entropy(z, y_train)
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
