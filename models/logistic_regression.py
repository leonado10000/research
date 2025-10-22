import torch
import torch.nn as nn
import wandb
import torchvision.datasets as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

WANDB_API_KEY = "f23a0839fa5de5085785818ab4c16be0f7be4ed9"

# wandb.init(
#     project="quickstart-playground",
#     config={
#         "lr": 0.01,
#         "epochs": 10,
#         "input_dim": 28*28,
#         "output_dim": 10,
#     }
# )

class LogReg(nn.Module):
    def __init__(self, input_dim):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_dim, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.linear(x)
        # out = torch.sigmoid(out)
        return out

transform = transforms.ToTensor()
training_data = tv.MNIST("./data/", train=True, transform=transform, download=True)
training_data = DataLoader(training_data, batch_size=64, shuffle=True)

model = LogReg(input_dim=28*28)
bce = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for X, y in training_data:
        y_pred = model(X)
        # print(y_pred.shape, y.shape)
        loss = bce(y_pred, y)
        loss.backward()
        sgd.step()
        # wandb.log({"epoch": epoch, "training-loss": loss.item()})

    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Sample Prediction: {loss.item()}")

testing_data = DataLoader(tv.MNIST("./data/", train=False, transform=transform, download=True), batch_size=64, shuffle=False)

total = 0
correct = 0
for X_test, y_test in testing_data:
    y_test_pred = model(X_test)
    test_loss = bce(y_test_pred, y_test)
    total += y_test.size(0)
    correct += (y_test_pred.argmax(dim=1) == y_test).sum().item()

# wandb.log({"test_loss": test_loss.item(), "test_accuracy": correct / total * 100})

print(f"Test Loss: {test_loss.item()}")
print(f"Test Accuracy: {correct / total * 100:.2f}%")