from module import *
from utils import *

class sgc(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512,  num_layers=2, dropout=0.5, in_drop=0):
        super(sgc, self).__init__()
        self.num_layers=num_layers
        if self.num_layers>1:
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins=(torch.nn.Linear(in_channels, out_channels))
            self.bns=(torch.nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.in_drop = in_drop

    def forward(self, x):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        if self.num_layers>1:
            for i, lin in enumerate(self.lins[:-1]):
                x = lin(x)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins(x)
        return x


class sgc_distill(nn.Module):
    def __init__(self, args, nfeat, nclass, hidden_channels,  num_layers, dropout, in_drop=0):
        super(sgc_distill, self).__init__()
        self.fc = torch.nn.ModuleList()
        for _ in range(args.num_max_hops):
            self.fc.append(sgc(nfeat, nclass,  hidden_channels,  num_layers, dropout, in_drop))
    def forward(self, num_hops, feature_list):
        return self.fc[num_hops-1](feature_list)



class sgc_distill_gate(nn.Module):
    def __init__(self, args, nfeat, nclass, hidden_channels,  num_layers, dropout, in_drop=0):
        super(sgc_distill_gate, self).__init__()
        self.fc = torch.nn.ModuleList()
        for _ in range(args.num_max_hops):
            self.fc.append(sgc(nfeat, nclass,  hidden_channels,  num_layers, dropout, in_drop))
        self.lr_att = nn.Linear(nclass, 1)

    def forward(self, num_hops, feature_list):
        return self.fc[num_hops-1](feature_list)
    
    def ensemble(self, feature_list, ensemble_layers):
        num_node=feature_list[0].shape[0]
        drop_features = [feature for feature in feature_list[-ensemble_layers:]]
        attention_scores = [torch.sigmoid(self.lr_att(x)).view(num_node,1) for x in drop_features]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W,1)
        x = torch.mul(drop_features[0], W[:,0].view(num_node,1)) 
        for i in np.arange(1, len(drop_features)):
            x += torch.mul(drop_features[i], W[:,i].view(num_node,1))
        return x         
