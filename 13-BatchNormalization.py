import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)  # Reproducibility


mnist_train = torchvision.datasets.MNIST(root="MNIST_data/", train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST(root="MNIST_data/", train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)

train_loader = DataLoader(mnist_train,
                          batch_size=100,
                          shuffle=True,
                          drop_last=True)
test_loader = DataLoader(mnist_test)


bn_linear1 = torch.nn.Linear(784, 32, bias=True)
bn_linear2 = torch.nn.Linear(32, 32, bias=True)
bn_linear3 = torch.nn.Linear(32, 10, bias=True)

nn_linear1 = torch.nn.Linear(784, 32, bias=True)
nn_linear2 = torch.nn.Linear(32, 32, bias=True)
nn_linear3 = torch.nn.Linear(32, 10, bias=True)

relu = torch.nn.ReLU()

bn1 = torch.nn.BatchNorm1d(32)
bn2 = torch.nn.BatchNorm1d(32)


bn_model = torch.nn.Sequential(bn_linear1, bn1, relu,
                               bn_linear2, bn2, relu,
                               bn_linear3)

nn_model = torch.nn.Sequential(nn_linear1, relu,
                               nn_linear2, relu,
                               nn_linear3)


criterion = torch.nn.CrossEntropyLoss()

bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=1e-3)
nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)

bn_training_loss = []
bn_test_loss = []

nn_training_loss = []
nn_test_loss = []


nb_epoch = 15
for epoch in range(1, nb_epoch + 1):
    print("epoch: {}".format(epoch))

    # Training
    bn_temp = []
    nn_temp = []

    bn_model.train()
    nn_model.train()
    for image, label in train_loader:
        # for thing in image:
        #     plt.imshow(thing.squeeze(), cmap="Greys", interpolation="nearest")
        #     plt.show(block=False)
        #     plt.pause(1)
        #     plt.close()

        image = image.view(-1, 28 * 28)

        bn_prediction = bn_model(image)
        nn_prediction = nn_model(image)

        bn_cost = criterion(bn_prediction, label)
        bn_temp.append(bn_cost.item())
        nn_cost = criterion(nn_prediction, label)
        nn_temp.append(nn_cost.item())

        bn_optimizer.zero_grad()
        bn_cost.backward()
        bn_optimizer.step()
        nn_optimizer.zero_grad()
        nn_cost.backward()
        nn_optimizer.step()

    bn_training_loss.append(sum(bn_temp) / len(bn_temp))
    nn_training_loss.append(sum(nn_temp) / len(nn_temp))

    # Test
    bn_temp = []
    nn_temp = []

    with torch.no_grad():
        bn_model.eval()
        nn_model.eval()
        for image, label in test_loader:
            image = image.view(-1, 28 * 28)

            bn_prediction = bn_model(image)
            nn_prediction = nn_model(image)

            bn_cost = criterion(bn_prediction, label)
            bn_temp.append(bn_cost.item())
            nn_cost = criterion(nn_prediction, label)
            nn_temp.append(nn_cost.item())

    bn_test_loss.append(sum(bn_temp) / len(bn_temp))
    nn_test_loss.append(sum(nn_temp) / len(nn_temp))


fig = plt.figure()
result = fig.add_subplot()

result.plot(bn_training_loss, label='bn_training')
result.plot(bn_test_loss, label='bn_test')
result.plot(nn_training_loss, label='nn_training')
result.plot(nn_test_loss, label='nn_test')

result.legend()
plt.title("Loss")
plt.show()
