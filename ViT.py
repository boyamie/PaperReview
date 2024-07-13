import os
from setuptools import find_packages
from setuptools import setup
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from tqdm import tqdm

def load_and_preprocess_data(data_dir, img_size=224, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

data_dir = '/Desktop/BOHYUN/0719'

# 데이터셋 로드 및 전처리
train_dataset, validation_dataset = load_and_preprocess_data(data_dir)

install_requires = [
    'absl-py',
    'aqtp!=0.1.1',  # https://github.com/google/aqt/issues/196
    'clu',
    'einops',
    'flax',
    'flaxformer @ git+https://github.com/google/flaxformer',
    'jax',
    'ml-collections',
    'numpy',
    'packaging',
    'pandas',
    'scipy',
     'torch', 
    'torchvision',
    'tqdm',
    'Pillow',  
    'albumentations',
    'tqdm',
]

tests_require = [
    'pytest',
]

__version__ = None

with open(os.path.join(here, 'version.py')) as f:
  exec(f.read(), globals())  # pylint: disable=exec-used

setup(
    name='vit_pytorch',
    version=__version__,
    description='Original PyTorch implementation of Vision Transformer models.',
    long_description=README,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    keywords='',
    author='Vision Transformer Authors',
    author_email='no-reply@google.com',
    url='https://github.com/google-research/vision_transformer',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=dict(test=tests_require),
)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT 모델 생성
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
model = model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# 모델 훈련
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)