import torch
import matplotlib.pyplot as plt


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([W, b], lr=1e-5)

cost_lst = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    Hypothesis = x_train.matmul(W) + b
    cost = torch.mean((Hypothesis - y_train) ** 2)  # MSE
    cost_lst.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


plt.title("cost")
plt.plot(cost_lst)
plt.show()
