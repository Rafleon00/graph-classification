"""
## Graph Neural Network Classification

### Objective:
Build a model that train and predicts based on Cora dataset
"""

# 1. Install dependencies
import torch
import pytorch_lightning as pl
import torch_geometric
import numpy as np
import argparse

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import LightningNodeData
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Optional

# 2. Set Defaults
NAME = "Cora"
BATCH_SIZE = 1
NUM_WORKERS = 2
HIDDEN_CHANNELS = 64
ATTENTION_HEADS = 4
DROPOUT = 0.5
LR = 1e-2
OPTIMIZER = "Adam"
LOSS = "cross_entropy"
pl.seed_everything(10)

# 3. Analyze Cora Dataset
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print(f'Dataset: {dataset}')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of node attributes: {dataset.num_node_features}')
print(f'Number of edge attributes: {dataset.num_edge_features}')

data = dataset[0] # Get the first graph object

print()
print(data)
print('===========================================================================================================')

# Information about the first graph object
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# 4. Create custom LightningDataModule for Cora dataset

class PlanetoidDataModule(pl.LightningDataModule):

  def __init__(self, args: argparse.Namespace = None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.batch_size = self.args.get("batch_size", BATCH_SIZE)
    self.num_workers = self.args.get("num_workers", NUM_WORKERS)

  def prepare_data(self):
    # Download Cora dataset
    Planetoid(root="data/Planetoid", name="Cora")

  def setup(self, stage: Optional[str] = None):
    # Preprocess Cora dataset
    self.dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())

  def train_dataloader(self):
      return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

  def test_dataloader(self):
    return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for multiple graph datasets, in case of only graph-dataset is defined to 1"
    )
    parser.add_argument(
        "--num_workers", type=int, default=NUM_WORKERS, help="Number of CPU cores for loading data"
    )
    return parser


# 5. Build GNN architecture based on multiple layers of GATs

class GNN(torch.nn.Module):

  def __init__(self, dataset, args: argparse.Namespace=None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.dropout = self.args.get("dropout", DROPOUT)
    self.input_dimension = dataset.num_features
    self.outpupt_dimension = dataset.num_classes
    self.hidden_channels = self.args.get("hidden_channels", HIDDEN_CHANNELS)
    self.attention_heads = self.args.get("attention_heads", ATTENTION_HEADS)
    self.gat1 = GATConv(self.input_dimension, self.hidden_channels, heads=self.attention_heads)
    self.gat2 = GATConv(self.attention_heads*self.hidden_channels, self.outpupt_dimension, heads=1)

  def forward(self, x, edge_index):
    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
    x = self.gat1(x, edge_index)
    x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
    x = self.gat2(x, edge_index)
    return x

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--hidden_channels", type=int, default=HIDDEN_CHANNELS, help="Hidden dimension of the GAT layers"
    )
    parser.add_argument(
        "--attention_heads", type=int, default=ATTENTION_HEADS, help="Number of attentio heads of GAT layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=DROPOUT, help="Training dropout"
    )
    return parser

# 6. Create custom LightningModule for the GNN model

class GNNLitModel(pl.LightningModule):

  def __init__(self, model, dataset, args: argparse.Namespace=None):
    super().__init__()
    self.args = vars(args) if args is not None else {}
    self.model = model
    self.dataset= dataset[0]
    self.lr = self.args.get("lr", LR)
    optimizer_class = self.args.get("optimizer", OPTIMIZER)
    self.optimizer = getattr(torch.optim, optimizer_class)
    loss_class = self.args.get("loss", LOSS)
    self.loss_fn = getattr(torch.nn.functional, loss_class)

  def forward(self, x, edge_index):
    return self.model(x, edge_index)

  def training_step(self, batch, batch_index):
    logits = self(batch.x, batch.edge_index)
    loss = self.loss_fn(logits[self.dataset.train_mask], batch.y[self.dataset.train_mask])
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_index):
    logits = self(batch.x, batch.edge_index)
    loss = self.loss_fn(logits[self.dataset.val_mask], batch.y[self.dataset.val_mask])
    self.log("val_loss", loss)
    return loss

  def test_step(self, batch, batch_idx):
    logits = self(batch.x, batch.edge_index)
    loss = self.loss_fn(logits[self.dataset.test_mask], batch.y[self.dataset.test_mask])
    self.log("test_loss", loss)
    return {"loss": loss,
            "predictions": logits[self.dataset.test_mask],
            "labels": batch.y[self.dataset.test_mask]}

  def make_predictions(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int().tolist()
    predictions = np.argmax(torch.stack(predictions), axis=1).tolist()
    return labels, predictions

  def test_epoch_end(self, outputs):
    labels, predictions = self.make_predictions(outputs)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    self.log("F1", f1)
    self.log("Recall", recall)
    self.log("Precision", precision)

  def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=self.lr)

  @staticmethod
  def add_to_argparse(parser):
    parser.add_argument(
        "--lr", type=float, default=LR, help="Learning rate of the training process"
    )
    parser.add_argument(
        "--optimizer", type=str, default=OPTIMIZER, help="Optimizer class from torch.optim"
    )
    parser.add_argument(
        "--loss", type=str, default=LOSS, help="Loss function from torch.functional"
    )
    return parser

# 7. Setup parser and run main
def _setup_parser():
  parser = argparse.ArgumentParser(add_help=False)
  # Add Lightning Trainer args
  parser = pl.Trainer.add_argparse_args(parser)
  # Add specific args
  parser = PlanetoidDataModule.add_to_argparse(parser)
  parser = GNN.add_to_argparse(parser)
  parser = GNNLitModel.add_to_argparse(parser)
  args = parser.parse_args()
  return args

def main():
  args = _setup_parser()
  datamodule = PlanetoidDataModule(args=args)
  datamodule.setup()
  gnn = GNN(dataset=datamodule.dataset, args=args)
  model = GNNLitModel(model=gnn, dataset=datamodule.dataset, args=args)
  early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
  model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
      filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min")
  callbacks = [early_stopping_callback, model_checkpoint_callback]
  trainer = pl.Trainer(accelerator="auto",
                       logger=pl.loggers.CSVLogger(save_dir="logs/"),
                       max_epochs=args.max_epochs,
                       callbacks=callbacks)
  # Training and validation
  trainer.fit(model=model, datamodule=datamodule)
  # Testing
  trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
  main()
