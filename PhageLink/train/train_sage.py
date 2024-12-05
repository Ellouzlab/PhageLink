import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.amp import autocast, GradScaler  # Updated import
from torch.profiler import profile, record_function, ProfilerActivity

# Define the custom EdgeDataset
class EdgeDataset(Dataset):
    def __init__(self, edge_label_index, edge_label_attr, edge_label):
        super(EdgeDataset, self).__init__()
        self.edge_label_index = edge_label_index
        self.edge_label_attr = edge_label_attr
        self.edge_label = edge_label

    def __len__(self):
        return self.edge_label_index.size(1)

    def __getitem__(self, idx):
        return {
            'edge_label_index': self.edge_label_index[:, idx],  # Shape: [2]
            'edge_label_attr': self.edge_label_attr[idx],      # Shape: [2]
            'edge_label': self.edge_label[idx]                 # Scalar
        }

class EdgePredictor(nn.Module):
    def __init__(self, device, node_feature_dim, edge_feature_dim, hidden_dim, num_classes):
        super(EdgePredictor, self).__init__()
        self.device = device
        self.sage1 = SAGEConv(node_feature_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, 32),  # Reduced hidden layer size
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, edge_label_index, edge_label_attr):
        # GraphSAGE Layers
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))

        # Get embeddings for edges we need to predict
        edge_src = x[edge_label_index[:, 0]]
        edge_dst = x[edge_label_index[:, 1]]

        # Concatenation of source and destination embeddings
        edge_emb = torch.cat([edge_src, edge_dst], dim=1)  # Shape: [batch_size, hidden_dim * 2]

        # Concatenate with edge attributes
        edge_features = torch.cat([edge_emb, edge_label_attr], dim=1)  # Shape: [batch_size, hidden_dim * 2 + edge_attr_dim]

        # Predict edge labels
        out = self.edge_mlp(edge_features)  # Shape: [batch_size, num_classes]
        return out

def train_graphsage_edge_model(hyper_adj_df, w_comb_adj_df, taxa_df, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure all DataFrames have the same genomes
    print("Aligning DataFrames...")
    genomes = list(set(hyper_adj_df.index) & set(w_comb_adj_df.index) & set(taxa_df.index))
    genomes.sort()
    num_nodes = len(genomes)

    # Map genomes to indices
    genome_to_idx = {genome: idx for idx, genome in enumerate(genomes)}

    # Create node features
    node_feature_dim = 16
    node_features = torch.randn(num_nodes, node_feature_dim, dtype=torch.float32)

    print("Building edge list without loops...")
    triu_indices = np.triu_indices(num_nodes, k=1)

    # Extract upper triangle values
    hyper_adj_values = hyper_adj_df.values[triu_indices]
    w_comb_adj_values = w_comb_adj_df.values[triu_indices]
    taxa_values = taxa_df.values[triu_indices]

    valid_mask = (~np.isnan(hyper_adj_values)) & (~np.isnan(w_comb_adj_values)) & (~np.isnan(taxa_values))
    print("done, now masking")
    edge_i = triu_indices[0][valid_mask]
    edge_j = triu_indices[1][valid_mask]
    feature1 = hyper_adj_values[valid_mask]
    feature2 = w_comb_adj_values[valid_mask]
    edge_labels = taxa_values[valid_mask].astype(int)

    # Duplicate edges for undirected graph
    edge_index = np.vstack((np.concatenate([edge_i, edge_j]),
                            np.concatenate([edge_j, edge_i])))

    edge_attrs = np.vstack((np.column_stack((feature1, feature2)),
                            np.column_stack((feature1, feature2))))

    edge_labels = np.concatenate([edge_labels, edge_labels])

    print("done, now converting to tensors")
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_labels = edge_labels

    num_edges = edge_labels.size(0)
    edge_indices = np.arange(num_edges)
    train_edges, test_edges = train_test_split(
        edge_indices, test_size=0.2, random_state=42, stratify=edge_labels.numpy()
    )

    print("done, now training")
    # Optimized Label Encoding using torch.unique
    unique_labels, inverse_indices = torch.unique(edge_labels, sorted=True, return_inverse=True)
    data.edge_labels = inverse_indices
    num_classes = unique_labels.size(0)
    print(f"Number of classes: {num_classes}")

    # Create masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[train_edges] = True
    test_mask[test_edges] = True
    print("done, now preparing data")

    # Prepare edge label indices and attributes for training
    data.train_edge_index = data.edge_index[:, train_mask]
    data.train_edge_label = data.edge_labels[train_mask]
    data.train_edge_attr = data.edge_attr[train_mask]

    # Prepare edge label indices and attributes for testing
    data.test_edge_index = data.edge_index[:, test_mask]
    data.test_edge_label = data.edge_labels[test_mask]
    data.test_edge_attr = data.edge_attr[test_mask]

    # Move all static data to device once
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.train_edge_index = data.train_edge_index.to(device)
    data.train_edge_label = data.train_edge_label.to(device)
    data.train_edge_attr = data.train_edge_attr.to(device)
    data.test_edge_index = data.test_edge_index.to(device)
    data.test_edge_label = data.test_edge_label.to(device)
    data.test_edge_attr = data.test_edge_attr.to(device)

    print("done, now scripting the model")
    # Instantiate the model
    hidden_dim = 16  # Reduced from 32
    model = EdgePredictor(
        device=device,  # Pass the device as a parameter
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_attr.size(1),
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)

    # Script the model for optimization
    model = torch.jit.script(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Initialize GradScaler for mixed precision with device specification
    scaler = GradScaler(device='cuda')  # Updated GradScaler initialization

    print("done, now preparing DataLoaders")
    # Create DataLoaders with optimizations
    train_dataset = EdgeDataset(
        edge_label_index=data.train_edge_index.cpu(),  # Data is already on device
        edge_label_attr=data.train_edge_attr.cpu(),
        edge_label=data.train_edge_label.cpu()
    )

    test_dataset = EdgeDataset(
        edge_label_index=data.test_edge_index.cpu(),
        edge_label_attr=data.test_edge_attr.cpu(),
        edge_label=data.test_edge_label.cpu()
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1024, 
        shuffle=True, 
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1024, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )

    print("done, now profiling and training loop")
    # enable profiling
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_training"):
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    edge_label_index = batch['edge_label_index'].to(device)    # [batch_size, 2]
                    edge_label_attr = batch['edge_label_attr'].to(device)      # [batch_size, 2]
                    edge_label = batch['edge_label'].to(device)                # [batch_size]
        
                    optimizer.zero_grad()
        
                    # Forward pass with autocast specifying device type
                    with autocast(device_type='cuda'):  # Updated autocast usage
                        out = model(
                            x=data.x,
                            edge_index=data.edge_index,
                            edge_attr=data.edge_attr,
                            edge_label_index=edge_label_index,
                            edge_label_attr=edge_label_attr
                        )
                        loss = criterion(out, edge_label)
        
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        
                    total_loss += loss.item() * edge_label.size(0)
        
                avg_loss = total_loss / len(train_loader.dataset)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    prof.export_chrome_trace("trace.json")
    """

    # Training loop with mixed precision
    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            edge_label_index = batch['edge_label_index'].to(device)    # [batch_size, 2]
            edge_label_attr = batch['edge_label_attr'].to(device)      # [batch_size, 2]
            edge_label = batch['edge_label'].to(device)                # [batch_size]

            optimizer.zero_grad()

            # Forward pass with autocast specifying device type
            with autocast(device_type='cuda'):  # Updated autocast usage
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_label_index=edge_label_index,
                    edge_label_attr=edge_label_attr
                )
                loss = criterion(out, edge_label)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * edge_label.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluation loop
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0

    # Initialize lists to collect all predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            edge_label_index = batch['edge_label_index'].to(device)
            edge_label_attr = batch['edge_label_attr'].to(device)
            edge_label = batch['edge_label'].to(device)

            # Forward pass with autocast specifying device type
            with autocast(device_type='cuda'):
                out = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_label_index=edge_label_index,
                    edge_label_attr=edge_label_attr
                )

            # Predictions
            pred = out.argmax(dim=1)

            # Collect predictions and true labels
            all_preds.append(pred.cpu())
            all_labels.append(edge_label.cpu())

            # Overall accuracy
            correct += (pred == edge_label).sum().item()
            total += edge_label.size(0)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute overall accuracy
    acc = correct / total
    print(f"Test Overall Accuracy: {acc:.4f}")

    # Compute per-class accuracy
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for cls in range(num_classes):
        # Find indices for each class
        cls_indices = (all_labels == cls)
        per_class_correct[cls] = (all_preds[cls_indices] == cls).sum().item()
        per_class_total[cls] = cls_indices.sum().item()

    # Avoid division by zero
    per_class_accuracy = torch.where(
        per_class_total > 0,
        per_class_correct / per_class_total,
        torch.zeros(1)
    )

    # Display per-class accuracy
    print("Per-Class Accuracy:")
    for cls in range(num_classes):
        class_name = unique_labels[cls].item()  # Retrieve original label
        print(f"Accuracy for class '{class_name}' (Label {cls}): {per_class_accuracy[cls]:.4f}")

    # Compute additional metrics using classification_report
    all_preds_np = all_preds.numpy()
    all_labels_np = all_labels.numpy()

    # Generate classification report
    report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names=unique_labels.numpy().astype(str),
        zero_division=0
    )

    print("Classification Report:")
    print(report)

    return model
