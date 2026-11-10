import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# 0. Настройки

BATCH_SIZE = 64
EPOCHS = 5
T = 50  # количество прогонов для MC Dropout
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {DEVICE}")

# 1. Загрузка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0 )

# 2. Модель: ResNet-18 + Dropout перед последним слоем
class SimpleMCResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # обучаем с нуля
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


model = SimpleMCResNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Обучение
print("Начинаем обучение...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Эпоха {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(trainloader):.4f}")

# 4. Функция MC Dropout предсказания

def mc_predict(model, dataloader, T=50):
    """
    Возвращает:
    mean_probs: (N, 10) средние вероятности
    uncertainty: (N,) дисперсия по классам (сумма)
    true_labels: (N,) истинные метки
    """
    model.train()  # включаем Dropout на тесте
    all_mean_probs = []
    all_uncertainty = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            batch_probs = []
            for _ in range(T):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs.unsqueeze(0))  # (1, batch, 10)

            batch_probs = torch.cat(batch_probs, dim=0)  # (T, batch, 10)
            mean_probs = batch_probs.mean(dim=0)  # (batch, 10)
            variance = batch_probs.var(dim=0).sum(dim=1)  # (batch,)

            all_mean_probs.append(mean_probs.cpu())
            all_uncertainty.append(variance.cpu())
            all_labels.append(labels)

    return (torch.cat(all_mean_probs),
            torch.cat(all_uncertainty),
            torch.cat(all_labels))


# 5. Предсказания на чистых тестовых данных
print("MC Dropout на чистых данных...")
clean_probs, clean_unc, clean_labels = mc_predict(model, testloader, T=T)

# Accuracy
clean_preds = clean_probs.argmax(dim=1)
clean_acc = (clean_preds == clean_labels).float().mean().item()
print(f"Accuracy на чистых данных: {clean_acc * 100:.2f}%")
print(f"Средняя неопределенность (чистые): {clean_unc.mean().item():.4f}")

# 6. Создаем "сдвинутые" данные
noise_level = 0.51  # сила шума
noisy_images_list = []
noisy_labels_list = []

for images, labels in testloader:
    noise = torch.randn_like(images) * noise_level
    noisy_images = torch.clamp(images + noise, -1, 1)  # клиппинг в диапазон [-1, 1]
    noisy_images_list.append(noisy_images)
    noisy_labels_list.append(labels)

noisy_dataset = torch.utils.data.TensorDataset(torch.cat(noisy_images_list), torch.cat(noisy_labels_list))
noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("MC Dropout на зашумленных данных...")
noisy_probs, noisy_unc, noisy_labels = mc_predict(model, noisy_loader, T=T)

noisy_preds = noisy_probs.argmax(dim=1)
noisy_acc = (noisy_preds == noisy_labels).float().mean().item()
print(f"Accuracy на зашумленных данных: {noisy_acc * 100:.2f}%")
print(f"Средняя неопределенность (шум): {noisy_unc.mean().item():.4f}")

# 7. ECE (10 корзин)
def compute_ece(probs, labels, n_bins=10):
    confidences = probs.max(dim=1).values  # уверенность модели (макс. вероятность)
    predictions = probs.argmax(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += torch.abs(avg_conf - avg_acc) * prop_in_bin
    return ece.item()


ece_clean = compute_ece(clean_probs, clean_labels)
ece_noisy = compute_ece(noisy_probs, noisy_labels)
print(f"ECE (чистые): {ece_clean:.4f}")
print(f"ECE (шум): {ece_noisy:.4f}")

# 8. Визуализация
# Гистограмма 1: неопределенность для правильных vs неправильных ответов (на чистых данных)
correct_mask = (clean_preds == clean_labels)
incorrect_mask = ~correct_mask

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(clean_unc[correct_mask].numpy(), bins=50, alpha=0.6, label='Правильные', color='green')
plt.hist(clean_unc[incorrect_mask].numpy(), bins=50, alpha=0.6, label='Неправильные', color='red')
plt.xlabel('Неопределенность (дисперсия)')
plt.ylabel('Количество примеров')
plt.title('Чистые данные: правильные vs неправильные')
plt.legend()

# Гистограмма 2: неопределенность на чистых vs шумных данных
plt.subplot(1, 2, 2)
plt.hist(clean_unc.numpy(), bins=50, alpha=0.6, label='Чистые', color='blue')
plt.hist(noisy_unc.numpy(), bins=50, alpha=0.6, label='Шумные', color='orange')
plt.xlabel('Неопределенность (дисперсия)')
plt.ylabel('Количество примеров')
plt.title('Чистые vs Шумные данные')
plt.legend()

plt.tight_layout()
plt.savefig('uncertainty_histograms.png', dpi=150)
print("Графики сохранены в файл uncertainty_histograms.png")
plt.show()