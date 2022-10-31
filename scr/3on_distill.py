from module import *
from utils import *
from pars import *
from model import *
from dataloader import * 
 

def model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader, feats_train_unlabel):
    best_epoch, best_val, count = 0, 0, 0

    for epoch in range(args.epochs+1):
        gc.collect()
        loss, acc, model = batch_train(args, model, feats_train, labels_train, loss_fcn, optimizer, train_loader, feats_train_unlabel)
        log = "Epoch {}, Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, loss, acc*100)
        if epoch % args.eval_every == 0:
            first_layer_val = batch_test_layers(args, model, feats_test, labels_test, val_loader, hop=1)
            log += "Epoch {}, Val {:.4f} ".format(epoch, first_layer_val)
            if first_layer_val > best_val:
                best_epoch = epoch
                best_val = first_layer_val
                acc_test = batch_test_layers(args, model, feats_test, labels_test, val_loader, hop=1)
                torch.save(model.state_dict(), args.on_distilled_model)
                count = 0
            else:
                count = count+args.eval_every
                if count >= args.patience:
                    break
            log += " Best Epoch {}, Val {:.4f}, Test {:.4f}, ".format(best_epoch, best_val, acc_test)
        if epoch % 1 == 0:
            print(log)
    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val, acc_test))
    return model



def batch_train(args, model, feats, labels, loss_fcn, optimizer, train_loader,feats_train_unlabel):

    model.train()
    y_true = []
    y_pred = []      
    for batch in train_loader:
        total_loss=0.
        batch_feats = [x[batch] for x in feats]
        batch_feats_train_unlabel=[x[batch] for x in feats_train_unlabel] if args.dataset not in ['pubmed'] else feats_train_unlabel
        batch_labels=labels[batch].to(args.device)
        model.to(args.device)

        soft_label_list=[]
        teacher_input=[]
        for hop in np.arange(1,args.num_max_hops+1):
            soft_label_list.append(model(hop, batch_feats_train_unlabel[hop].to(args.device)) )
            teacher_input.append(model(hop, batch_feats[hop].to(args.device)) )

        soft_label=model.ensemble(soft_label_list,  ensemble_layers=args.ensemble_layers)
        soft_label=soft_label.to(args.device)   

        teacher_logits=model.ensemble(teacher_input,  ensemble_layers=args.ensemble_layers)
        teacher_loss=loss_fcn(teacher_logits, batch_labels)

        for hop in np.arange(1,2):
            output_att=model(hop,batch_feats[hop].to(args.device))
            output_att_distill=model(hop,batch_feats_train_unlabel[hop].to(args.device))
            kl = kd_loss_function(output_att_distill, soft_label, args)         
            loss_train = ((1-args.lambda_on)*loss_fcn(output_att, batch_labels) + args.lambda_on*kl)
            total_loss += loss_train 
            y_true.append(batch_labels)
            y_pred.append(output_att.argmax(dim=-1))            

        total_loss += teacher_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss = total_loss / args.num_max_hops

    evaluator=get_acc_evaluator()         
    acc=(evaluator(torch.cat(y_pred, dim=0), torch.cat(y_true)))
    return loss, acc, model


def kd_loss_function(output, target_output,args):
    target_output= target_output/args.temperature_on
    target_output = torch.softmax(target_output, dim=1)
    output = output / args.temperature_on
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    loss_kd=loss_kd*(args.temperature_on**2)
    return loss_kd

def batch_test_layers(args, model, feats, labels, loader,hop):
    model.to(args.device)
    preds = []
    true = []
    evaluator=get_acc_evaluator()
    with torch.no_grad():    
        model.eval()
        for batch in loader:
            batch_feats = feats[hop][batch].to(args.device)
            output_att=model(hop, batch_feats) if model._get_name()!='sgc' else  model(batch_feats)
            true.append(labels[batch].to(args.device))
            preds.append(torch.argmax(output_att, dim=-1))
        true = torch.cat(true)
        preds = torch.cat(preds, dim=0)
        res = evaluator(preds, true)
    return res


def test_model(model, feats_test, labels_test, val_loader, test_loader):
    with open(args.on_model_result,'a') as f:
        f.writelines("Online distillation results: ") 
        f.write('\n')   

    for num_hops in np.arange(1,args.num_max_hops+1):
        val_acc = batch_test_layers(args, model, feats_test, labels_test, val_loader, num_hops)
        test_acc = batch_test_layers(args, model, feats_test, labels_test, test_loader, num_hops)
        with open(args.on_model_result,'a') as f:
            f.writelines("Layer: ")
            f.writelines(str(num_hops))
            f.writelines("    Val Accuracy: ")
            f.writelines(str(round(val_acc,5))) 
            f.writelines("    Test Accuracy: ")
            f.writelines(str(round(test_acc,5)))            
            f.write('\n') 



if __name__ == "__main__":
    args=pars()
    args=args_setting(args)

    g, labels, train_nid, val_nid, test_nid, unlabeled_nid, args = load_dataset(args)
    feats_test, labels_test = data_transductive(args, g, labels, train_nid, val_nid, test_nid, unlabeled_nid)    
    feats_train, labels_train, train_nid = data_inductive(args, g, labels, val_nid, test_nid)
    feats_train_unlabel = data_inductive_distill(args, g, labels, val_nid, test_nid)


    train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid), len(train_nid)+len(val_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    model_initial = sgc_distill(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    model_initial.load_state_dict(torch.load(args.off_distilled_model,map_location=torch.device('cpu' if args.device=='cpu' else args.device)))
    set_seed(args.seed)
    model = sgc_distill_gate(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)   
    model = initial_model_multi(args, model, model_initial)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc[0].parameters():
        param.requires_grad = True      
    for param in model.lr_att.parameters():
        param.requires_grad = True 

    loss_fcn =nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= args.lr,weight_decay=args.weight_decay) 
    model = model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader, feats_train_unlabel)


    model = sgc_distill_gate(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)   
    model.load_state_dict(torch.load(args.on_distilled_model,map_location=torch.device('cpu' if args.device=='cpu' else args.device)))
    for hop in np.arange(2, args.num_max_hops+1):
        model = initial_model_layer(model, model_initial, hop)
    torch.save(model.state_dict(), args.on_distilled_model)
    test_model(model, feats_test, labels_test, val_loader, test_loader)
