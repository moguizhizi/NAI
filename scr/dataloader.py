from module import *
from utils import *
from dataloader import * 

def load_dataset(args):
    dataset = dgl.data.FlickrDataset(raw_dir=args.data_dir)    
    g = dataset[0]
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']
    labels = g.ndata['label'].long()
    g.ndata['labels']= labels
    idx=torch.tensor(np.arange(0,len(labels))).long()
    unlabeled_mask= train_mask^(~(val_mask | test_mask))
    
    train_nid=idx[train_mask.bool()]
    val_nid=idx[val_mask.bool()]
    test_nid=idx[test_mask.bool()]
    unlabeled_nid=idx[unlabeled_mask.bool()]   

    g=dgl.remove_self_loop(g)
    g=dgl.add_self_loop(g)
    g = node_norm(g)

    print(f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n")
    args.in_size=int(g.ndata['feat'].shape[1])
    args.n_classes=int(labels.max()+1)
    args.n_nodes=g.num_nodes()
    args.n_edges=g.number_of_edges()
    args.out_n_nodes=test_nid.shape[0]
    gc.collect()
    return g, labels, train_nid, val_nid, test_nid, unlabeled_nid, args


def node_norm(sg):
    degs = sg.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    sg.ndata['norm'] = norm.unsqueeze(1)
    return sg


def data_transductive(args, g, labels, train_nid, val_nid, test_nid, unlabeled_nid):
    feats = neighbor_norm_features(g, args)
    gc.collect()
    feat = []
    for  x in feats:
        feat.append( torch.cat((x[train_nid], x[val_nid], x[test_nid], x[unlabeled_nid]), dim=0))
    labels=torch.cat([labels[train_nid], labels[val_nid], labels[test_nid], labels[unlabeled_nid]])
    return feat, labels

def neighbor_norm_features(g, args):

    g.ndata["feat_0"] = (g.ndata["feat"])
    for hop in range(1, args.num_max_hops + 1):
        g.ndata[f"feat_{hop}"] = g.ndata[f"feat_{hop-1}"] * g.ndata['norm']
        g.update_all(fn.copy_u(f"feat_{hop}", "msg"), fn.sum("msg", f"feat_{hop}"))
        g.ndata[f"feat_{hop}"] = g.ndata[f"feat_{hop}"] * g.ndata['norm']
    x = []
    for hop in range(args.num_max_hops + 1):
        if args.dataset == "ogbn-mag":
            x.append(g.ndata.pop(f"feat_{hop}")[g.ndata["target_mask"]])
        else:
            x.append(g.ndata.pop(f"feat_{hop}"))
    return x

def data_inductive(args, g, labels, val_nid, test_nid):

    training_mask =inductive_mask(labels.size(0), val_nid, test_nid)
    sg = dgl.node_subgraph(g, training_mask)
    sg = node_norm(sg)
    sg = normalize_feature(sg)
    feats = neighbor_norm_features(sg, args)
    train_mask = sg.ndata['train_mask'].bool()
    labels = sg.ndata['labels']
    idx=torch.tensor(np.arange(0,len(labels))).long()
    train_nid=idx[train_mask]
    feat = []
    for _, x in enumerate(feats):
        feat.append(x[train_mask])
    labels=labels[train_mask]
    gc.collect()
    return feat, labels, train_nid

def data_inductive_distill(args, g, labels, val_nid, test_nid):
    training_mask =inductive_mask(labels.size(0), val_nid, test_nid)
    sg = dgl.node_subgraph(g, training_mask)
    sg = node_norm(sg)
    sg = normalize_feature(sg)
    feats = neighbor_norm_features(sg, args)
    labels = sg.ndata['labels']
    gc.collect()
    return feats

def normalize_feature(g):
    mx=g.ndata['feat']
    mx=mx-mx.min()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    g.ndata['feat']=torch.tensor(mx)
    return g


def features_norm(g, args, test_nid):
    """
    Compute multi-hop neighbor-averaged node features
    """
    test_nid=test_nid[0]
    sample_time=[]
    t0=time.time()
    nodes_list=[]
    edge_list=[]
    nodes_list.append(test_nid)
    edge_list.append(0)
    for _ in range(args.Tmax):
        m=g.in_edges(nodes_list[-1],form='all')      
        a=torch.unique(m[0].to("cuda:0"),sorted=False) # cuda is faster
        b=torch.unique(m[1].to("cuda:0"),sorted=False)        
        nodes_list.append(torch.unique(torch.cat((a,b),dim=-1),sorted=False).cpu())
        edge_list.append(m[2])
    sample_time=(time.time()-t0)

    feat_process_time=[]
    t0=time.time()  
    g.ndata["feat_0"] = g.ndata["feat"]
    # full propagation 
    for hop in range(1, args.Tmax + 1):
        g.ndata[f"feat_{hop}"] = g.ndata[f"feat_{hop-1}"] * g.ndata['norm']
        g.update_all(fn.copy_u(f"feat_{hop}", "msg"), fn.sum("msg", f"feat_{hop}"))
        g.ndata[f"feat_{hop}"] = g.ndata[f"feat_{hop}"] * g.ndata['norm']
        feat_process_time.append(time.time()-t0)

    x = []
    for hop in range(args.Tmax + 1):
        x.append(g.ndata.pop(f"feat_{hop}"))

    return x, edge_list, nodes_list, sample_time, feat_process_time