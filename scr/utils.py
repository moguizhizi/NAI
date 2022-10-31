from module import *

def out_flops_batch_A(args, edge_list, node_list, out_num_list): 

    flops_out=[]
    flops_feat=0.
    output_node=0
    in_feat = args.in_size 
    n_dim=args.hidden_channels
    args.evl_node_num=int(sum(out_num_list))
    in_feat_dif = round(args.in_size * args.dif_dim_ratio)

    entity = edge_list[-1].size(0)
    flops_feat+=(2*entity)/1e9    # normalize A 
    flops_Ainf=(args.n_nodes+ args.n_nodes+args.evl_node_num+in_feat_dif*(2*args.n_nodes-1)+in_feat_dif*args.evl_node_num)/1e9# A_inf*H
    for i in np.arange(len(edge_list)-1, 0, -1):
        entity = edge_list[i].size(0)
        out_num = node_list[i-1].shape[0]
        flops_feat+=in_feat*(2*entity-out_num)/1e9   # AH 
        if i !=0:
            flops_feat+=(args.evl_node_num-output_node)*(3*in_feat*args.dif_dim_ratio-1) /1e9  # difference
        output_node+=int(out_num_list[len(node_list)-1-i])
        this_layer_out=int(out_num_list[len(node_list)-1-i])
        if this_layer_out!=0:
            flops_out.append(this_layer_out*(flops_feat))
    return flops_Ainf, np.sum(flops_out)/args.evl_node_num

def original_flops_batch_A(args, edge_list, node_list):  
    flops_feat=0.
    in_feat = args.in_size
    n_dim=args.hidden_channels
    entity = edge_list[-1].size(0)
    flops_feat+=(2*entity)/1e9 # normalize A 
    for i in np.arange(len(edge_list)-1, 0, -1):
        entity = edge_list[i].size(0)
        out_num = node_list[i-1].shape[0]
        flops_feat+=in_feat*(2*entity-out_num)/1e9   # AH  
    if args.num_layers==2:
        flops_classifier=(out_num*n_dim*(2*in_feat-1) /1e9 ) +(out_num*args.n_classes*(2*n_dim-1) /1e9 ) 
    elif args.num_layers==1:
        flops_classifier=out_num*args.n_classes*(2*in_feat-1) /1e9    
    elif args.num_layers==3:
        flops_classifier=(out_num*n_dim*(2*in_feat-1) /1e9 ) + (out_num*n_dim*(2*n_dim-1) /1e9 ) +(out_num*args.n_classes*(2*n_dim-1) /1e9 ) 
    return flops_feat, flops_classifier

def cal_inf_A(args, g, test_nid):
    test_nid=test_nid[0]
    degree=g.in_degrees().float()
    node_sum=args.n_nodes
    edge_sum=args.n_edges
    features=g.ndata['feat']
    if args.dif_dim_ratio<1:
        diffrence_dim_num=round(args.dif_dim_ratio*int(features.shape[1]))
        diffrence_dim=np.array([False]*int(features.shape[1]))
        diffrence_dim[:diffrence_dim_num]=True
        random.seed(0)
        random.shuffle(diffrence_dim)
        diffrence_dim=torch.tensor(diffrence_dim)
        features=features[:,diffrence_dim]
    t0=time.time()
    row_sum = degree + 1
    d_inv = torch.float_power(row_sum, 0.5).flatten().float()
    d_inv_target=d_inv.reshape(-1,1)[test_nid]/(2*edge_sum+node_sum)
    norm_a_inf = torch.mm(d_inv.reshape(1,-1),features)
    norm_a_inf = torch.spmm(d_inv_target, norm_a_inf)
    inf_a_time=round(time.time()-t0,4)
    return norm_a_inf, inf_a_time*1e3


def record_result_batch(args, results, Ts, Tmin, Tmax, val_acc_ori, val_acc_p, out_num_list, 
                flops_feat_ori_batch, flops_classifier,  flops_Ainf, flops_feat_pro_batch,
                testset_acc):

    flops_batch_ori=(flops_feat_ori_batch+ flops_classifier)
    flops_batch_pro=(flops_Ainf+flops_feat_pro_batch+ flops_classifier)
    speed_flops_x_batch=round(((flops_batch_pro)*100/flops_batch_ori),3)
    speed_flops_p_batch=round(((flops_batch_ori-flops_batch_pro)*100/flops_batch_ori),3)

    print_file_batch(args, val_acc_ori, flops_batch_ori, out_num_list, val_acc_p, speed_flops_x_batch, Tmin, Tmax)
    results.append([Ts, Tmin, Tmax, out_num_list, 
    val_acc_ori, val_acc_p,
    [flops_batch_ori, flops_batch_pro, speed_flops_x_batch, speed_flops_p_batch],
    testset_acc])   
    with open(args.hyper_settings, 'wb') as f:
        pickle.dump(results, f) 
    return results


def print_file_batch(args, test_acc_ori, flops_batch_ori, out_num_list, test_acc_p, speed_flops_x_batch, Tmin, Tmax):
    print("Val Accuracy: {:.4f}".format(test_acc_ori))
    print("Original FLOPs: {:.3f} GB".format(flops_batch_ori))

    print('_'*10)
    print("Val Accuracy: {:.4f}".format(test_acc_p))    
    print("Ts:",args.Ts)
    print("Tmin:", Tmin)
    print("Tmax:", Tmax)
    print("Exit samples:", list(np.array(out_num_list).reshape(-1).astype(int) ))    

    print("FLOPs: {:.2f} %".format(speed_flops_x_batch))
    print('*'*70)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)



def get_acc_evaluator():
    evaluator = Evaluator(name='ogbn-arxiv')
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]



def initial_model(model_distll, model, num_hops):
    trained_layer_dict=model_distll.fc[num_hops-1].state_dict()
    model_dict=model.state_dict()
    for k, _ in trained_layer_dict.items():
        if model_dict[k].shape == trained_layer_dict[k].shape: 
            trained_layer_dict[k]=model_dict[k]

    model_distll.fc[num_hops-1].load_state_dict(trained_layer_dict)
    return model_distll 

def initial_model_multi(args, model_distll, model):
    for num_hops in np.arange(1,args.num_max_hops+1):  
        trained_layer_dict=model_distll.fc[num_hops-1].state_dict()
        model_dict=model.fc[num_hops-1].state_dict()
        for k, _ in trained_layer_dict.items():
            if model_dict[k].shape == trained_layer_dict[k].shape: 
                trained_layer_dict[k]=model_dict[k]

        model_distll.fc[num_hops-1].load_state_dict(trained_layer_dict)
    return model_distll 


def initial_model_layer(model_distll, model, num_hops):
    trained_layer_dict=model_distll.fc[num_hops-1].state_dict()
    model_dict=model.fc[num_hops-1].state_dict()
    for k, _ in trained_layer_dict.items():
        if model_dict[k].shape == trained_layer_dict[k].shape: 
            trained_layer_dict[k]=model_dict[k]
    model_distll.fc[num_hops-1].load_state_dict(trained_layer_dict)
    return model_distll 


def inductive_mask(node_num, val_nid, test_nid):
    training_mask  = torch.Tensor([True]*node_num).bool()
    training_mask[val_nid]=False
    training_mask[test_nid]=False
    return training_mask

