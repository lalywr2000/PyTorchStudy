import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])  # one-hot vector: index of 1

x_test = torch.FloatTensor([[2, 1, 1],
                            [3, 1, 2],
                            [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])  # one-hot vector: index of 1


model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

training_loss = []
test_loss = []


nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    # Train
    prediction = model(x_train)

    cost = F.cross_entropy(prediction, y_train)
    training_loss.append(cost.item())

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # Test
    prediction = model(x_test)

    cost = F.cross_entropy(prediction, y_test)
    test_loss.append(cost.item())


fig = plt.figure()
result = fig.add_subplot()

result.plot(training_loss, label='Training')
result.plot(test_loss, label='Test')

result.legend()
plt.title("loss")
plt.show()
