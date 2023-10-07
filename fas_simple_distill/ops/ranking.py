import torch
from torch.autograd import Variable
import torch.nn.functional as F

def ranking_loss(score, targets):
    if torch.cuda.is_available():
        loss = Variable(torch.zeros(1).cuda())
    else:
        loss = Variable(torch.zeros(1))
    batch_size = score.size(0)

    if torch.cuda.is_available():
        data_type = torch.cuda.FloatTensor
    else:
        data_type = torch.FloatTensor
    for i in range(targets.shape[1]):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(data_type)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size

def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)