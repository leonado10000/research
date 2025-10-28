from models.logistic_regression import LogReg
import torch
import torchvision.datasets as tv
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb

model = LogReg(input_dim=28*28)
epochs = 10

try:
    wandb.init(
        project="quickstart-playground",
        config={
            "input_dim": model.input_dim,
            "output_dim": model.output_dim,
            "model": model.model_name,
        }
    )
except Exception as e:
    print(f"Wandb init failed: {e}")

test_data = tv.MNIST("./data/", train=False, transform=transforms.ToTensor(), download=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

total = 0
correct = 0
for X_test, y_test in test_data_loader:
    y_test_pred = model(X_test)
    test_loss = model.loss_function(y_test_pred, y_test)
    total += y_test.size(0)
    correct += (y_test_pred.argmax(dim=1) == y_test).sum().item()

try:
    wandb.log({"test_loss": test_loss.item(), "test_accuracy": correct / total * 100})
except Exception as e:
    print(f"Wandb log failed: {e}")

print(f"Test Loss: {test_loss.item()}")
print(f"Test Accuracy: {correct / total * 100:.2f}%")