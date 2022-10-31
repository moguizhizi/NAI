from module import *
from utils import *
from pars import *
from model import *
from dataloader import * 
 

def Inference_ori(args, adj, feature, test_nid, test_loader, labels):
    set_seed(args.seed)
    model = sgc(args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    model.load_state_dict(torch.load(args.base_model, map_location=torch.device('cpu')))    
    model = model.to(args.device)
    evaluator = get_acc_evaluator()


    preds_list=[]
    logits_list=[]
    labels_list=[]
    classification_time=[] 
    feat_time=[]
    sample_time=[]
    flops_feat_list=[]
    flops_classifier_list=[]
    counter=0
    for batch in test_loader:
        counter+=1
        if counter>3:
            continue
        test_mask = np.array([False]*(adj.shape[0]))
        test_mask[batch]=True
        nodes_list=[]
        adj_list=[]
        nodes_list.append(test_mask)
        sam_time=0
        for k in range(args.num_max_hops):
            t0=time.time()
            adj_t=adj[nodes_list[-1]]
            m = np.array(adj_t.sum(axis=0)>0).reshape(-1)
            adj_list.append(adj[nodes_list[k],:][:,m]) 
            sam_time+=time.time()-t0
            nodes_list.append(m)
        sample_time.append(sam_time)

        start = time.time()
        feat = feature[nodes_list[-1]]
        for i in np.arange(len(adj_list)-1, -1, -1):
            feat=adj_list[i]@feat
        feat_time.append((time.time()-start))  
        start = time.time()
        with torch.no_grad():
            model.eval()
            logits=model(torch.tensor(feat).to(args.device))
            preds=torch.argmax(logits, dim=-1)
        classification_time.append((time.time()-start))            
        preds_list.append(preds)
        logits_list.append(torch.softmax(logits,dim=1))
        labels_list.append(labels[batch])

        flops_feat, flops_classifier = original_flops(args, nodes_list, adj_list)
        flops_feat_list.append(flops_feat) 
        flops_classifier_list.append(flops_classifier) 

    label = torch.cat(labels_list)
    pred = torch.cat(preds_list, dim=0) 
    logit = torch.cat(logits_list, dim=0)    
    acc = evaluator(pred, label)

    return acc, np.mean(feat_time)*1e3, np.mean(classification_time)*1e3, np.mean(sample_time)*1e3, np.mean(flops_feat_list), np.mean(flops_classifier_list)


def Inference_pro(args, adj,  feature, test_nid, norm_a_inf, test_loader, labels):

    model = sgc_distill_gate(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    model.load_state_dict(torch.load(args.on_distilled_model))
    model = model.to(args.device) 
    model.eval()
    if args.dif_dim_ratio<1:
        diffrence_dim_num=round(args.dif_dim_ratio*args.in_size)
        diffrence_dim=np.array([False]*args.in_size)
        diffrence_dim[:diffrence_dim_num]=True
        random.seed(0)
        random.shuffle(diffrence_dim)
    else:
        diffrence_dim=np.array([True]*args.in_size)

    norm_a_inf_all=np.zeros((g.num_nodes(),round(args.dif_dim_ratio*args.in_size)))
    norm_a_inf_all[test_nid]=norm_a_inf
    out_num_total=None
    labels_list=[]
    preds_list=[]
    feat_process_time_list=[]
    sample_time=[]
    logits_list=[]
    flops_Ainf_list=[]
    flops_feat_pro_list=[]
    counter=0
    for batch in test_loader:
        counter+=1
        if counter>3:
            continue 
        test_mask = np.array([False]*(adj.shape[0]))
        test_mask[batch]=True
        nodes_list=[]
        index=[]
        adj_list=[]        
        nodes_list.append(test_mask)
        index.append(test_mask[test_mask])
        sam_time=0
        for k in range(args.Tmax):
            t0=time.time()
            adj_t=adj[nodes_list[-1]]
            m = np.array(adj_t.sum(axis=0)>0).reshape(-1)
            adj_list.append(adj[nodes_list[k],:][:,m])
            sam_time+=time.time()-t0
            nodes_list.append(m)
            index.append(test_mask[m])
        sample_time.append(sam_time)

        feat = feature[nodes_list[-1]]
        out_num=[]
        feat_process_time=0
        mask_before = np.array([False]*(batch.shape[0]))
        pre_label=torch.zeros(batch.shape[0], args.n_classes).to(args.device)
        start=time.time()
        for hop in range(1, args.Tmax + 1):
            if hop<=args.Tmin:
                out_num.append(0)
                feat=adj_list[-hop]@feat

            elif hop < args.Tmax:
                
                feat=adj_list[-hop]@feat
                feat_out=feat[index[-hop-1]]
                dist = np.linalg.norm((feat_out[:,diffrence_dim] - norm_a_inf_all[batch]), ord=2, axis=1)
                if args.Ts_weight==None:
                    args.Ts_weight=float(dist.max())
                mask = (dist<(args.Ts*args.Ts_weight))
                mask[mask_before]=False
                mask_before[mask]=True
                out_num.append(int(sum(mask)))
                if out_num[-1] != 0: 
                    logits =feat_out[mask] 
                    with torch.no_grad():  
                        pre_label[mask] = model(hop,torch.Tensor(logits).to(args.device))    
                    feat_process_time += (time.time()-start)  *int(out_num[-1])     

            elif hop == args.Tmax:
                feat=adj_list[-hop]@feat
                feat_out=feat[index[-hop-1]]
                mask = np.array([True]*batch.shape[0])
                mask[mask_before]=False
                out_num.append(int(mask.sum()))
                if out_num[-1] != 0: 
                    logits =feat_out[mask]
                    with torch.no_grad():  
                        pre_label[mask] = model(hop,torch.Tensor(logits).to(args.device))      
                    feat_process_time += (time.time()-start)  *int(out_num[-1])
                    
        preds_list.append(torch.argmax(pre_label, dim=-1))
        logits_list.append(torch.softmax(pre_label,dim=1))
        labels_list.append(labels[batch])
        feat_process_time_list.append(feat_process_time/batch.shape[0])

        flops_Ainf, flops_feat_pro = out_flops(args, adj_list, nodes_list, out_num)  
        flops_Ainf_list.append(flops_Ainf)
        flops_feat_pro_list.append(flops_feat_pro)


    evaluator = get_acc_evaluator()
    labels = torch.cat(labels_list)
    preds = torch.cat(preds_list, dim=0)
    logit = torch.cat(logits_list, dim=0)
    acc = evaluator(preds, labels)

    return acc, np.mean(feat_process_time_list)*1e3, np.mean(sample_time)*1e3,np.mean(flops_Ainf_list),np.mean(flops_feat_pro_list)



def get_adj_feat(g):
    row, col = g.edges()
    N = g.num_nodes()
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout='csr')
    feat= g.ndata['feat']
    return adj, feat


def original_flops(args,  nodes_list, adj_list):   
    flops_feat=0.
    in_feat = args.in_size
    n_dim=args.hidden_channels
    entity = adj_list[-1].nnz
    flops_feat+=(2*entity)/1e6 # normalize A 
    for i in np.arange(len(nodes_list)-1, 0, -1):
        entity = adj_list[i-1].nnz
        out_num = nodes_list[i-1].sum()
        flops_feat+=in_feat*(2*entity-out_num)/1e6   # AH     
    if args.num_layers==2:
        flops_classifier=(out_num*n_dim*(2*in_feat-1) /1e6 ) +(out_num*args.n_classes*(2*n_dim-1) /1e6 ) 
    elif args.num_layers==1:
        flops_classifier=out_num*args.n_classes*(2*in_feat-1) /1e6    
    elif args.num_layers==3:
        flops_classifier=(out_num*n_dim*(2*in_feat-1) /1e6 ) + (out_num*n_dim*(2*n_dim-1) /1e6 ) +(out_num*args.n_classes*(2*n_dim-1) /1e6 ) 
    return flops_feat, flops_classifier


def out_flops(args, adj_list, node_list, out_num_list):
    flops_out=[]
    flops_feat=0.
    output_node=0
    in_feat = args.in_size 
    args.evl_node_num=int(sum(out_num_list))
    in_feat_dif = round(args.in_size * args.dif_dim_ratio)
    entity = adj_list[-1].nnz
    flops_feat+=(2*entity)/1e6 # normalize A 
    flops_Ainf=(args.n_nodes+ args.n_nodes+args.evl_node_num+in_feat_dif*(2*args.n_nodes-1)+in_feat_dif*args.evl_node_num)/1e6# A_inf*H
    for i in np.arange(args.Tmax, 0, -1):
        entity = adj_list[i-1].nnz
        out_num = node_list[i-1].sum()
        flops_feat+=in_feat*(2*entity-out_num)/1e6   # AH 
        if i !=1:
            flops_feat+=(args.evl_node_num-output_node)*(3*in_feat*args.dif_dim_ratio-1) /1e6  # difference
        output_node+=int(out_num_list[args.Tmax-i])
        this_layer_out=int(out_num_list[args.Tmax-i])
        if this_layer_out!=0:
            flops_out.append(this_layer_out*(flops_feat))

    return flops_Ainf, np.sum(flops_out)/args.evl_node_num








if __name__ == "__main__":
    args=pars()
    args.gpu=-1
    args=args_setting(args)

    g, labels, train_nid, val_nid, test_nid, unlabeled_nid, args = load_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_nid, batch_size=1000000, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_nid, batch_size=args.test_node_num, shuffle=False, drop_last=True)
    adj, feat= get_adj_feat(g)

    ori_time=[]
    pro_time=[]
    c_time_ori=[]
    f_process_time_pro=[]
    d_time_ori=[]
    d_time=[]
    ori_flops=[]
    pro_flops=[]
    for iter in range(3):
        _, feat_time, classification_time_ori, dataload_time_ori, flops_feat_ori, flops_classifier = Inference_ori(args, adj, feat, test_nid, test_loader, labels)
        norm_a_inf, inf_a_time = cal_inf_A(args, g, [test_nid] )
        _, feat_process_time_pro, dataload_time, flops_Ainf, flops_feat_pro  = Inference_pro(args, adj,  feat, test_nid, norm_a_inf, test_loader, labels) 
       

        ori_flops.append([flops_feat_ori+flops_classifier,   
            flops_feat_ori, 
            flops_classifier])
        pro_flops.append([flops_Ainf+flops_feat_pro+flops_classifier, 
            flops_Ainf,    
            flops_feat_pro,    
            flops_classifier])
        ori_time.append([feat_time+classification_time_ori+dataload_time_ori,    
            dataload_time_ori,    
            feat_time,    
            classification_time_ori])
        pro_time.append([feat_process_time_pro+inf_a_time+dataload_time,
            inf_a_time,    
            dataload_time,    
            feat_process_time_pro-classification_time_ori,     
            classification_time_ori])



    # total
    ori_time_a=[np.mean([i[0] for i in ori_time]), np.std([i[0] for i in ori_time])]
    pro_time_a=[np.mean([i[0] for i in pro_time]), np.std([i[0] for i in pro_time])]
    ori_flops_a=np.mean([i[0] for i in ori_flops])
    pro_flops_a=np.mean([i[0] for i in pro_flops]) 
    # time
    ori_time_samplegraph=[np.mean([i[1] for i in ori_time]), np.std([i[1] for i in ori_time])]
    ori_time_feat=[np.mean([i[2] for i in ori_time]), np.std([i[2] for i in ori_time])]
    ori_time_classification=[np.mean([i[3] for i in ori_time]), np.std([i[3] for i in ori_time])]

    pro_time_Ainf=[np.mean([i[1] for i in pro_time]), np.std([i[1] for i in pro_time])]
    pro_time_samplegraph=[np.mean([i[2] for i in pro_time]), np.std([i[2] for i in pro_time])]
    pro_time_feat=[np.mean([i[3] for i in pro_time]), np.std([i[3] for i in pro_time])]
    pro_time_classification=[np.mean([i[4] for i in pro_time]), np.std([i[4] for i in pro_time])]

    # Flops
    ori_flops_feat=np.mean([i[1] for i in ori_flops])
    ori_flops_classification=np.mean([i[2] for i in ori_flops])

    pro_flops_Ainf=np.mean([i[1] for i in pro_flops])
    pro_flops_feat=np.mean([i[2] for i in pro_flops])
    pro_flops_classification=np.mean([i[3] for i in pro_flops]) 


    log = "Batch size: {:.1f} ".format(args.test_node_num) +'\n'
    log += "Layers: {:.1f} ".format(args.num_max_hops) +'\n'

    log += "Ori:    "+'\n'
    log += "   Time (ms/node): {:.4f} + {:.4f}".format(ori_time_a[0],ori_time_a[1])
    log += "   Flops (M/node): {:.4f} ".format(ori_flops_a)+'\n'
    log += "Ori time details: "+'\n'
    log += "   Sample Time (ms/node): {:.4f} + {:.4f}".format(ori_time_samplegraph[0],ori_time_samplegraph[1])+'\n'
    log += "   Feature Time (ms/node): {:.4f} + {:.4f} ".format(ori_time_feat[0],ori_time_feat[1])+'\n'
    log += "   classification Time (ms/node): {:.4f} + {:.4f} ".format(ori_time_classification[0],ori_time_classification[1])+'\n'
    log += "Ori Flops details: "+'\n'
    log += "   Feature Flops (M/node): {:.4f} ".format(ori_flops_feat)+'\n'
    log += "   classification Flops (M/node): {:.4f} ".format(ori_flops_classification)+'\n'+'\n'

    log += "Pro:    "+'\n'
    log += "   Time (ms/node): {:.4f} + {:.4f}".format(pro_time_a[0],pro_time_a[1])
    log += "   Flops (M/node): {:.4f} ".format(pro_flops_a)  +'\n'

    log += "Pro time details: "+'\n'
    log += "   Ainf Time (ms/node): {:.4f} + {:.4f} ".format(pro_time_Ainf[0],pro_time_Ainf[1])+'\n'
    log += "   Sample Time (ms/node): {:.4f} + {:.4f} ".format(pro_time_samplegraph[0],pro_time_samplegraph[1])+'\n'
    log += "   Feature Time (ms/node): {:.4f} + {:.4f} ".format(pro_time_feat[0],pro_time_feat[1])+'\n'
    log += "   classification Time (ms/node): {:.4f} + {:.4f} ".format(pro_time_classification[0],pro_time_classification[1])+'\n'
    log += "Pro Flops details: "+'\n'
    log += "   Ainf Flops (M/node): {:.4f} ".format(pro_flops_Ainf)+'\n'
    log += "   Feature Flops (M/node): {:.4f} ".format(pro_flops_feat)+'\n'
    log += "   classification Flops (M/node): {:.4f} ".format(pro_flops_classification)+'\n'+'\n'


    log += "Total: "+'\n' 
    log += "Flops Speed up: {:.3f} X".format(ori_flops_a/pro_flops_a) +'\n' 
    log += "Time Speed up: {:.3f} X".format(ori_time_a[0]/pro_time_a[0]) +'\n' 
    log += "Flops Speed up (only feature processing): {:.3f}X".format(ori_flops_feat/pro_flops_feat) +'\n' 
    log += "Time Speed up (only feature processing): {:.3f}X".format((ori_time_samplegraph[0]+ori_time_feat[0])/(pro_time_feat[0]+pro_time_samplegraph[0])) +'\n' 
    log += '\n'
    log += '\n'
    log += '\n'
    print(log)
    with open(args.inference_result,'a') as f:
        f.writelines(log)



