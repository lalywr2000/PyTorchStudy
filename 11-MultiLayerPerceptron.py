import torch
import matplotlib.pyplot as plt


x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

linear1 = torch.nn.Linear(2, 10, bias=True)
linear2 = torch.nn.Linear(10, 10, bias=True)
linear3 = torch.nn.Linear(10, 10, bias=True)
linear4 = torch.nn.Linear(10, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

cost_lst = []


nb_epoch = 10000
for epoch in range(1, nb_epoch + 1):
    prediction = model(x)

    cost = torch.nn.functional.binary_cross_entropy(prediction, y)
    cost_lst.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


plt.title("loss")
plt.plot(cost_lst)
plt.show()
