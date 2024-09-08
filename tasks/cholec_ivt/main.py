import logging

import ivtmetrics
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CholecT50
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(filename='/mnt/training.log', encoding='utf-8', level=logging.INFO, filemode='w')

# Create a logger
logger = logging.getLogger()

# Prevent log messages from child loggers from propagating to the parent logger
logger.propagate = False

model = timm.create_model("timm/davit_tiny.msft_in1k", pretrained=True, num_classes=100)

data_config = timm.data.resolve_model_data_config(model)
model_transforms = timm.data.create_transform(**data_config, is_training=True)

data_builder = CholecT50("./CholecT50", "cholect50-challenge")#, augmentation_list=['original'])
train, val, test = data_builder.build()


# # Comment out image return in T50 __getitem__ to accelerate calculation of labels
# train_loader = DataLoader(train, batch_size=10_000, shuffle=False, num_workers=8)
# val_loader = DataLoader(val, batch_size=10_000, shuffle=False, num_workers=8)
# test_loader = DataLoader(test, batch_size=10_000, shuffle=False, num_workers=8)
# c = torch.zeros([100])
# for d in [train_loader, val_loader, test_loader]:
#     for lab in tqdm(d):
#         c += torch.sum(lab[0], axis=0)
# total_len = len(train) + len(val) + len(test)
# pos_weight = (total_len - c)/c
# torch.save(pos_weight, './pos_weight.pt')
pos_weight = torch.load('./pos_weight.pt')


# Create your DataLoaders
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

recognize = ivtmetrics.Recognition(num_class=100)

# Training loop
num_epochs = 100
train_lossi = []
train_losse = []
val_losse = []
accuracyi = []
best_acc = 0
print(f'Training starting for {num_epochs} epochs...')
logger.info(f'Training starting for {num_epochs} epochs...')
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        # only considering ivt label
        labels = labels[0].to(device).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_lossi.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    recognize.reset_global()
    with torch.no_grad():
        val_loss = 0.0
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels[0].to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Store predictions and true labels for precision and recall calculation
            np_out = (torch.sigmoid(outputs)>0.5).int().cpu().numpy()
            np_labels = labels.cpu().numpy()
            val_preds.extend(np_out)
            val_targets.extend(np_labels)
            recognize.update(np_labels, np_out)
            
        recognize.video_end()
            
    accuracy = accuracy_score(val_targets, val_preds)
    precision = precision_score(val_targets, val_preds, average='samples', zero_division=1)
    recall = recall_score(val_targets, val_preds, average='samples', zero_division=1)

    tl = train_loss/len(train_loader)
    vl = val_loss/len(val_loader)
    train_losse.append(tl)
    val_losse.append(vl)
    accuracyi.append(accuracy*100)

    if accuracy > best_acc:
        torch.save(model.state_dict(), './best_davit_tiny.pth')
        
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tl:.5f}, Valid Loss: {vl:.5f}")
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy*100}, Precision: {precision*100}, Recall: {recall*100}")
    results_ivt = recognize.compute_video_AP('ivt')
    print(f"Epoch {epoch+1}/{num_epochs}, Triple meanAP: {results_ivt['mAP']*100}")
    torch.save(model.state_dict(), f'./davit_tiny/epoch{epoch:03d}.pth')

    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tl:.5f}, Valid Loss: {vl:.5f}")
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy*100}, Precision: {precision*100}, Recall: {recall*100}")
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Triple meanAP: {results_ivt['mAP']*100}")
    logger.info("*"*20)

logger.info('Training Completed...')