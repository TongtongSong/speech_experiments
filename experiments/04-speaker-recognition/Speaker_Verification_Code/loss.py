import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(Softmax, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embeddings, labels):
        # ==========> code <===========


        # ==========> code <===========
        acc = self.accuracy(logits, labels)
        return loss, acc
    
    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total


class Amsoftmax(nn.Module):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(Amsoftmax, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.margin = margin
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embeddings, labels):
        logits = F.linear(F.normalize(embeddings), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, labels.view(-1,1), self.margin)
        m_logits = self.s * (logits - margin)
        loss = F.cross_entropy(m_logits, labels)

        acc = self.accuracy(logits, labels)
        return loss, acc

    def accuracy(self, logit, label):
        answer = (torch.max(logit, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = logit.size(0)

        return answer / n_total