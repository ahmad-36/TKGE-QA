
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import QuadrupleDataLoader as dl

class TTransE(nn.Module):
    def __init__(self, nrRelations, nrEntities, nrTimes, dimEmbedding, margin=1.0):
        super().__init__()
        self.nrRelations = nrRelations
        self.nrEntities = nrEntities
        self.nrTimes = nrTimes
        self.dimEmbedding = dimEmbedding
        self.margin = margin

        self.entities = nn.Embedding(self.nrEntities, self.dimEmbedding)
        nn.init.xavier_uniform_(self.entities.weight)
        self.relations = nn.Embedding(self.nrRelations, self.dimEmbedding)
        nn.init.xavier_uniform_(self.relations.weight)
        self.times = nn.Embedding(self.nrTimes, self.dimEmbedding)
        nn.init.xavier_uniform_(self.times.weight)

    def generate_negative_samples(self, input):
        subject_ids = input[0]
        object_ids = input[2]
        batch_size = subject_ids.size(0)
        device = subject_ids.device

        subject_offset = torch.randint(1, self.nrEntities, (batch_size,), device=device)
        object_offset  = torch.randint(1, self.nrEntities, (batch_size,), device=device)

        negative_subject_ids = (subject_ids + subject_offset) % self.nrEntities
        negative_object_ids  = (object_ids + object_offset) % self.nrEntities
        return negative_subject_ids, negative_object_ids

    def forward(self, input):
        subjects  = self.entities(input[0])
        relations = self.relations(input[1])
        objects   = self.entities(input[2])
        times     = self.times(input[3])
        # plausibility score: higher is better
        return -torch.norm(subjects + relations + times - objects, p=2, dim=1)

    def compute_loss(self, input, positive_scores):
        negative_subject_ids, negative_object_ids = self.generate_negative_samples(input)

        neg_input_h = [negative_subject_ids, input[1], input[2], input[3]]
        neg_input_t = [input[0], input[1], negative_object_ids, input[3]]

        negative_scores = torch.cat([self(neg_input_h), self(neg_input_t)], dim=0)
        positive_scores_expanded = positive_scores.repeat(2)

        loss_fn = nn.MarginRankingLoss(margin=self.margin, reduction="mean")
        labels = torch.ones_like(positive_scores_expanded)
        return loss_fn(positive_scores_expanded, negative_scores, labels)

def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for sample in loader:
            sample = [s.to(device) for s in sample]
            scores = model(sample)
            loss = model.compute_loss(sample, scores)
            total += float(loss.item())
            n += 1
    return total / max(n, 1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train_dataset = dl.ICEWSData("train", True, base_dir=".", valid_ratio=0.05, seed=13)
    valid_dataset = dl.ICEWSData("valid", True, base_dir=".", valid_ratio=0.05, seed=13)

    nrEntities  = len(train_dataset.entity2id)
    nrRelations = len(train_dataset.relation2id)
    nrTimes     = len(train_dataset.time2id)

    dimEmbedding = 200
    margin = 1.0

    model = TTransE(nrRelations, nrEntities, nrTimes, dimEmbedding, margin).to(device)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.07)

    best_valid = float("inf")
    best_path = "ttranse_icews14_best.pt"

    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for sample in train_loader:
            sample = [s.to(device) for s in sample]
            optimizer.zero_grad()
            scores = model(sample)
            loss = model.compute_loss(sample, scores)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1

        train_loss = total / max(n, 1)
        valid_loss = evaluate_loss(model, valid_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "nrEntities": nrEntities,
                "nrRelations": nrRelations,
                "nrTimes": nrTimes,
                "dimEmbedding": dimEmbedding,
                "margin": margin,
                "id_map_paths": {
                    "entity2id": "icews14/entity2id.txt",
                    "relation2id": "icews14/relation2id.txt",
                    "time2id": "icews14/time2id.txt",
                },
                "valid_ratio": 0.05,
                "seed": 13,
            }
            torch.save(ckpt, best_path)
            print(f"  Saved best checkpoint -> {best_path} (valid_loss={best_valid:.6f})")

if __name__ == "__main__":
    main()
