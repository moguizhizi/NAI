from module import *
from utils import *
from pars import *
from model import *
from dataloader import * 


def model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader, feats_train_unlabel,soft_label):
    best_epoch, best_val, best_test, count  = 0, 0, 0, 0 
    for epoch in range(args.epochs+1):
        gc.collect()

        loss, acc, model = batch_train(args, model, feats_train[args.num_hops], labels_train, loss_fcn, optimizer, train_loader, feats_train_unlabel[args.num_hops],soft_label)
        log = "Epoch {}, Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, loss, acc)
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = batch_test(args, model, feats_test[args.num_hops], labels_test, val_loader)
            log += "Epoch {}, Val: {:.4f}, ".format(epoch, acc)
            if acc > best_val:
                best_epoch = epoch
                best_val = acc
                best_test = batch_test(args, model, feats_test[args.num_hops], labels_test, test_loader)
                torch.save(model.state_dict(), args.off_distilled_model)
                count = 0
            else:
                count = count+args.eval_every
                if count >= args.patience:
                    break
            log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(
                best_epoch, best_val, best_test)
        if epoch % 1 == 0:
            print(log)

    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val, best_test))

    with open(args.off_model_result,'a') as f:
        f.writelines("Layer: ")  
        f.write(str(args.num_hops) ) 
        f.writelines("    Val Accuracy: ")
        f.write(str(round(best_val,5)) )        
        f.writelines("    Test Accuracy: ")
        f.write(str(round(best_test,5)) )
        f.write('\n')   
    return model



def batch_train(args, model, feats, labels, loss_fcn, optimizer, train_loader,feats_train_unlabel, soft_label):
    model.train()

    total_loss, iter_num= 0, 0
    y_true, y_pred= [], []  
    for batch in train_loader:
        batch_feats = feats[batch].to(args.device)
        feats_train_unlabel=feats_train_unlabel.to(args.device)
        labels=labels.to(args.device)
        soft_label=soft_label.to(args.device)
        model=model.to(args.device)
        output_att = model(args.num_hops,batch_feats)
        output_att_distill = model(args.num_hops,feats_train_unlabel)
        y_true.append(labels[batch])

        y_pred.append(output_att.argmax(dim=-1))
        kl = kd_loss_function(output_att_distill, soft_label, args)         
        loss_train = ((1-args.lambda_off)*loss_fcn(output_att, labels[batch]) + args.lambda_off*kl)

        total_loss += loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num += 1
    loss = total_loss / iter_num

    evaluator=get_acc_evaluator()
    acc = evaluator(torch.cat(y_pred, dim=0), torch.cat(y_true))
    return loss, acc, model

def kd_loss_function(output, target_output,args):
    target_output= target_output/args.temperature_off
    target_output = torch.softmax(target_output, dim=1)

    output = output / args.temperature_off
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    loss_kd=loss_kd*(args.temperature_off**2)
    return loss_kd
    
def batch_test(args, model, feats, labels, loader):

    
    model.to(args.device)
    preds, true= [], []
    evaluator=get_acc_evaluator()
    with torch.no_grad():    
        for batch in loader:
            batch_feats = feats[batch].to(args.device)
            true.append(labels[batch].to(args.device))
            model.eval()
            output_att=model(args.num_hops, batch_feats) if model._get_name()!='sgc' else  model(batch_feats)
            preds.append(torch.argmax(output_att, dim=-1))
        true = torch.cat(true)
        preds = torch.cat(preds, dim=0)
        res = evaluator(preds, true)

    return res

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

    set_seed(args.seed)
    model_teacher = sgc(args.in_size,  args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers)
    model_teacher.load_state_dict(torch.load(args.base_model,map_location=torch.device('cpu' if args.device=='cpu' else args.device)))
    
    soft_label_list=[]
    with torch.no_grad():
        model_teacher.eval()
        soft_label=model_teacher(feats_train_unlabel[-1])
    
    with open(args.off_model_result,'a') as f:
        f.writelines("Offline distillation results: ")
        f.write('\n') 

    for num_hops in np.arange(1,args.num_max_hops+1):
        args.num_hops=num_hops
        set_seed(args.seed)
        model = sgc_distill(args, args.in_size, args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)

        if args.num_hops>1:
            model.load_state_dict(torch.load(args.off_distilled_model))

        model = initial_model(model, model_teacher, args.num_hops)
        if num_hops==args.num_max_hops:
            torch.save(model.state_dict(), args.off_distilled_model)
            break

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc[num_hops-1].parameters():
            param.requires_grad = True  

        loss_fcn =nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=args.weight_decay)
        model = model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader, feats_train_unlabel,soft_label)
