from module import *
from utils import *
from pars import *
from model import *
from dataloader import * 


def Inference_ori(args, feats, labels):
    set_seed(args.seed)
    model = sgc(args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    model.load_state_dict(torch.load(args.base_model, map_location=torch.device('cpu' if args.device=='cpu' else args.device)))
    model = model.to(args.device)
    start = time.time()
    with torch.no_grad():
        model.eval()
        logits=model(feats.to(args.device))
    evaluator = get_acc_evaluator()        
    preds=torch.argmax(logits, dim=-1)
    acc = evaluator(preds, labels)
    end = time.time()
    classification_time = round(end - start,2)
    return acc, classification_time


def Inference_pro(args, g, feats, test_nid, labels, norm_a_inf, feat_time, model):
    model = model.to(args.device) 
    model.eval() 
    feat_process_time = 0
    pre_label=torch.zeros(test_nid.size(0), args.n_classes)
    if args.dif_dim_ratio<1:
        diffrence_dim_num=round(args.dif_dim_ratio*args.in_size)
        diffrence_dim=np.array([False]*args.in_size)
        diffrence_dim[:diffrence_dim_num]=True
        random.seed(0)
        random.shuffle(diffrence_dim)
        diffrence_dim=torch.tensor(diffrence_dim) 
    else:
        diffrence_dim=torch.tensor([True]*args.in_size)
    mask_before = torch.Tensor([False]*(test_nid.size(0))).bool()
    out_num=[]
  
    for hop in range(1, args.Tmax + 1):
        if hop<=args.Tmin:
            out_num.append(0)
        elif hop < args.Tmax:
            dist = torch.linalg.norm((feats[hop][:,diffrence_dim] - norm_a_inf), ord=2, dim=1)
            if args.Ts_weight==None:
                args.Ts_weight=dist.max()
            mask = (dist<(args.Ts*args.Ts_weight)).masked_fill_(mask_before, False)
            mask_before.masked_fill_(mask, True)  
            out_num.append(int(mask.sum()))
            if out_num[-1] != 0: 
                logits=feats[hop][mask]  
                with torch.no_grad(): 
                    model.eval()  
                    pre_label[mask] = model(hop,logits)
                feat_process_time += (feat_time[hop-1])  *int(out_num[-1])
        elif hop == args.Tmax: 
            mask = torch.Tensor([True]*test_nid.size(0)).bool()
            mask.masked_fill_(mask_before, False)
            out_num.append(int(mask.sum()))
            if out_num[-1] != 0: 
                logits=feats[hop][mask]   
                with torch.no_grad():
                    model.eval()  
                    pre_label[mask] = model(hop,logits) 
                feat_process_time += (feat_time[hop-1])  *int(out_num[-1])
    preds=torch.argmax(pre_label, dim=-1)
    evaluator = get_acc_evaluator()
    acc = evaluator(preds, labels)
    feat_process_time = round(feat_process_time/args.out_n_nodes,2)

    return acc, feat_process_time, out_num





if __name__ == "__main__":
    args=pars()
    args.gpu=-1
    args=args_setting(args)

    g, labels, train_nid, val_nid, test_nid, unlabeled_nid, args = load_dataset(args)
    feats_val, edge_list, nodes_list, sample_time, feat_process_time_ori = features_norm(g, args, [val_nid] )
    val_acc_ori, classification_time_ori = Inference_ori(args, feats_val[-1][val_nid], labels[val_nid])
    flops_feat_ori_batch, flops_classifier = original_flops_batch_A(args, edge_list, nodes_list)

    model = sgc_distill_gate(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    model.load_state_dict(torch.load(args.on_distilled_model))

    results=[]
    for Tmax in np.arange(args.num_max_hops, 0, -1):
        args.Tmax=Tmax
        feats_val, edge_list, nodes_list, sample_time, feat_process_time_ori = features_norm(g, args, [val_nid] )
        norm_a_inf, inf_a_time = cal_inf_A(args, g, [val_nid])
        for Tmin in [0,1,2]: 
            args.Tmin =Tmin
            if args.Tmin>=args.Tmax:
                continue
            for Ts in np.arange(0, 0.2, 0.0005):
                if Ts>=0.2-0.0005:
                    Ts=1
                if args.Tmin ==args.Tmax-1 and Ts > 0 :
                    continue                        
                args.Ts=round(Ts,5)
                val_acc_p, feat_process_time_pro, out_num_list = Inference_pro(args, g, [x[val_nid] for x in feats_val], val_nid, labels[val_nid], norm_a_inf, feat_process_time_ori, model)
                flops_Ainf, flops_feat_pro_batch = out_flops_batch_A(args, edge_list, nodes_list, out_num_list) 
                norm_a_inf_test, _ = cal_inf_A(args, g, [test_nid])
                test_acc_p, _, _ = Inference_pro(args, g, [x[test_nid] for x in feats_val], test_nid, labels[test_nid], norm_a_inf_test, feat_process_time_ori, model)
                results = record_result_batch(args, results, Ts, Tmin, Tmax, val_acc_ori, val_acc_p, out_num_list, 
                                        flops_feat_ori_batch, flops_classifier,  flops_Ainf, flops_feat_pro_batch, test_acc_p)



