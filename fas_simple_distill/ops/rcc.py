import torch
import torch.nn as nn



class rcc_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def l2_norm(input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def count_cos(self, f, g):
        numerator = torch.dot(f, g)
        denominator = torch.dot(self.l2_norm(f, axis=1), self.l2_norm(g, axis=1))

        return torch.div(numerator, denominator)

    def forward(self, x, sub_lbl):
        start = True
        batch_size = int(x.size(dim=0))

        for idx in range(batch_size):
            x_data = x[idx]
            sub_lbl_data = sub_lbl[idx]

            pos_idx = torch.where(sub_lbl == sub_lbl_data)
            neg_idx = torch.where(sub_lbl != sub_lbl_data)
            pos_data = x[pos_idx]
            neg_data = x[neg_idx]
            avg_pos_data = pos_data.mean()
            avg_neg_data = neg_data.mean()

            # neg_size = neg_data.size(0)

            # neg_start = True
            # neg_cos = 0

            # for neg_idx in range(neg_size):
            #     neg_sample = neg_data[neg_idx]
                
            #     if neg_start:
            #         neg_cos = self.count_cos(x_data, neg_sample)
            #         neg_start = False
            #     else:
            #         neg_cos += self.count_cos(x_data, neg_sample)
        
            # neg_cos = torch.div(neg_cos, neg_size)
            
            lbl = torch.randn(sub_lbl.size()).to(self.device)
            lbl[pos_idx] = 0
            lbl[neg_idx] = 1

            if start:
                rcc_loss = torch.log(torch.exp(torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_neg_data.view(-1, 1))) + torch.exp(torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_pos_data.view(-1, 1)))) - (lbl * torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_neg_data.view(-1, 1))) - ((1 - lbl) * torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_pos_data.view(-1, 1)))
                start = False
            else:
                rcc_loss = torch.log(torch.exp(torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_neg_data.view(-1, 1))) + torch.exp(torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_pos_data.view(-1, 1)))) - (lbl * torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_neg_data.view(-1, 1))) - ((1 - lbl) * torch.matmul(torch.transpose(x_data.unsqueeze(0), dim0=0, dim1=1), avg_pos_data.view(-1, 1)))

        # loss = torch.div(rcc_loss, x.size(dim=0))
        return rcc_loss.mean()
