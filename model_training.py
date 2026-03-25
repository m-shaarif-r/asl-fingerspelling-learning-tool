import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import zipfile

# ------------------ CONFIG ------------------
zip_path = "ASL_Processed_Images.zip"
extract_path = "ASL_Processed_Images"

model_save_path = "Model/pytorch_model.pth"

batch_size = 32
epochs = 5
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ EXTRACT ZIP (if needed) ------------------
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

# ------------------ FIND TRAIN / TEST FOLDERS ------------------
train_dir = os.path.join(extract_path, "train")
test_dir  = os.path.join(extract_path, "test")

if not os.path.exists(train_dir):
    inner = os.listdir(extract_path)[0]
    train_dir = os.path.join(extract_path, inner, "train")
    test_dir  = os.path.join(extract_path, inner, "test")

print("Train dir:", train_dir)
print("Test dir:", test_dir)

# ------------------ TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------------------ LOAD DATASETS ------------------
full_train = datasets.ImageFolder(root=train_dir, transform=transform)
full_test  = datasets.ImageFolder(root=test_dir, transform=transform)

# ------------------ KEEP ONLY A–Z ------------------
letter_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

valid_indices = [
    i for i, cls in enumerate(full_train.classes)
    if cls in letter_classes
]

# IMPORTANT: rebuild label mapping correctly
class_to_idx = {cls: i for i, cls in enumerate(letter_classes)}

# ------------------ FILTERED DATASET (keeps only A-Z and remaps labels) ------------------
train_indices = [
    i for i, (_, label) in enumerate(full_train.samples)
    if full_train.classes[label] in letter_classes
]

test_indices = [
    i for i, (_, label) in enumerate(full_test.samples)
    if full_test.classes[label] in letter_classes
]

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, class_to_idx):
        self.base = base_dataset
        self.indices = list(indices)
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base.samples[real_idx]
        image = self.base.loader(path)
        if self.base.transform is not None:
            image = self.base.transform(image)
        cls = self.base.classes[label]
        new_label = self.class_to_idx[cls]
        return image, new_label


train_dataset = FilteredDataset(full_train, train_indices, class_to_idx)
test_dataset  = FilteredDataset(full_test, test_indices, class_to_idx)

# ------------------ DATALOADERS ------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = letter_classes
num_classes = len(class_names)

print("Classes:", class_names)

# ------------------ MODEL ------------------
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ------------------ LOSS + OPTIMIZER ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------ TRAINING LOOP ------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ------------------ EVALUATION ------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

# ------------------ SAVE MODEL ------------------
os.makedirs("Model", exist_ok=True)
torch.save(model, model_save_path)

print("Training complete. Model saved.")