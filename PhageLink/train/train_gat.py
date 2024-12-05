import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
            'edge_label_index': self.edge_label_index[:, idx],
            'edge_label_attr': self.edge_label_attr[idx],
            'edge_label': self.edge_label[idx]
        }

# Define the EdgePredictor model
class EdgePredictor(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, num_classes):
        super(EdgePredictor, self).__init__()
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, edge_label_index, edge_label_attr):
        # Move tensors to device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # GAT Layers
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Get node embeddings for the edge_label_index
        edge_src = x[edge_label_index[0]]
        edge_dst = x[edge_label_index[1]]

        # Element-wise multiplication
        edge_emb = edge_src * edge_dst

        # Concatenate with edge_label_attr
        edge_features = torch.cat([edge_emb, edge_label_attr.to(device)], dim=1)

        # Edge classification MLP
        out = self.edge_mlp(edge_features)
        return out

# Define the training function
def train_gat_edge_model(hyper_adj_df, w_comb_adj_df, taxa_df, num_epochs=10):
    global device  # Make device accessible inside the model
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure all DataFrames have the same genomes (nodes)
    print("Aligning DataFrames...")
    genomes = list(set(hyper_adj_df.index) & set(w_comb_adj_df.index) & set(taxa_df.index))
    genomes.sort()
    num_nodes = len(genomes)

    # Map genomes to indices
    genome_to_idx = {genome: idx for idx, genome in enumerate(genomes)}

    # Create node features (randomly initialized)
    node_feature_dim = 16
    node_features = torch.randn(num_nodes, node_feature_dim)

    # Build edge list with features and labels without loops
    print("Building edge list without loops...")
    # Get upper triangle indices
    triu_indices = np.triu_indices(num_nodes, k=1)

    # Extract upper triangle values
    hyper_adj_values = hyper_adj_df.values[triu_indices]
    w_comb_adj_values = w_comb_adj_df.values[triu_indices]
    taxa_values = taxa_df.values[triu_indices]

    # Create a mask for valid edges (non-NaN in all matrices)
    valid_mask = (~np.isnan(hyper_adj_values)) & (~np.isnan(w_comb_adj_values)) & (~np.isnan(taxa_values))

    # Apply mask to indices and values
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

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    # Create Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_labels = edge_labels

    # Split edges into train and test sets
    num_edges = edge_labels.size(0)
    edge_indices = np.arange(num_edges)
    train_edges, test_edges = train_test_split(
        edge_indices, test_size=0.2, random_state=42, stratify=edge_labels
    )

    # Prepare edge label encoder
    label_encoder = LabelEncoder()
    data.edge_labels = torch.tensor(label_encoder.fit_transform(edge_labels), dtype=torch.long)
    num_classes = len(label_encoder.classes_)

    # Create masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[train_edges] = True
    test_mask[test_edges] = True

    # Prepare edge label indices and attributes for training
    data.train_edge_index = data.edge_index[:, train_mask]
    data.train_edge_label = data.edge_labels[train_mask]
    data.train_edge_attr = data.edge_attr[train_mask]

    # Prepare edge label indices and attributes for testing
    data.test_edge_index = data.edge_index[:, test_mask]
    data.test_edge_label = data.edge_labels[test_mask]
    data.test_edge_attr = data.edge_attr[test_mask]

    # Move data to CPU (we'll move batches to GPU)
    data = data.to('cpu')

    # Instantiate the model
    model = EdgePredictor(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_attr.size(1),
        hidden_dim=32,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Create DataLoaders
    train_dataset = EdgeDataset(
        edge_label_index=data.train_edge_index,
        edge_label_attr=data.train_edge_attr,
        edge_label=data.train_edge_label
    )

    test_dataset = EdgeDataset(
        edge_label_index=data.test_edge_index,
        edge_label_attr=data.test_edge_attr,
        edge_label=data.test_edge_label
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Training loop
    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            edge_label_index = batch['edge_label_index'].to(device)
            edge_label_attr = batch['edge_label_attr'].to(device)
            edge_label = batch['edge_label'].to(device)

            # Forward pass
            out = model(
                x=data.x.to(device),
                edge_index=data.edge_index.to(device),
                edge_attr=data.edge_attr.to(device),
                edge_label_index=edge_label_index,
                edge_label_attr=edge_label_attr
            )

            # Compute loss
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * edge_label.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluation loop
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            edge_label_index = batch['edge_label_index'].to(device)
            edge_label_attr = batch['edge_label_attr'].to(device)
            edge_label = batch['edge_label'].to(device)

            # Forward pass
            out = model(
                x=data.x.to(device),
                edge_index=data.edge_index.to(device),
                edge_attr=data.edge_attr.to(device),
                edge_label_index=edge_label_index,
                edge_label_attr=edge_label_attr
            )

            # Predictions
            pred = out.argmax(dim=1)
            correct += (pred == edge_label).sum().item()
            total += edge_label.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

    return model
