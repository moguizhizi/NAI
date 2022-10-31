from module import *
from utils import *
from pars import *
from model import *
from dataloader import * 


def model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader):
    best_epoch, best_val, best_test, count  = 0, 0, 0, 0
    for epoch in range(args.epochs+1):
        gc.collect()

        loss, acc, model = batch_train(args, model, feats_train[args.num_hops], labels_train, loss_fcn, optimizer, train_loader)
        log = "Epoch {}, Train loss: {:.4f}, Train acc: {:.4f} ".format(epoch, loss, acc)
        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = batch_test(args, model, feats_test[args.num_hops], labels_test, val_loader)
            log += "Epoch {}, Val {:.4f} ".format(epoch, acc)
            if acc > best_val:
                best_epoch = epoch
                best_val = acc
                best_test = batch_test(args, model, feats_test[args.num_hops], labels_test, test_loader)
                if args.num_hops==args.num_max_hops:
                    torch.save(model.state_dict(), args.base_model)
                count = 0
            else:
                count = count+args.eval_every
                if count >= args.patience:
                    break
            log += "Best Epoch {},Val {:.4f}, Test {:.4f} ".format(best_epoch, best_val, best_test)
        if epoch % 1 == 0:
            print(log)

    print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))

    with open(args.initial_model_result,'a') as f:
        f.writelines("Layer: ")  
        f.write(str(args.num_hops) ) 
        f.writelines("   Val Accuracy: ")  
        f.write(str(round(best_val, 5)) )        
        f.writelines("   Test Accuracy: ")  
        f.write(str(round(best_test, 5)) )
        f.write('\n')   




def batch_train(args, model, feats, labels, loss_fcn, optimizer, train_loader):
    model.train()
    
    total_loss, iter_num= 0, 0
    y_true, y_pred= [], []
    for batch in train_loader:
        batch_feats = feats[batch].to(args.device)
        labels=labels.to(args.device)
        model=model.to(args.device)

        output_att = model(batch_feats)
        y_true.append(labels[batch])
        y_pred.append(output_att.argmax(dim=-1))

        loss_train = loss_fcn(output_att, labels[batch])
        total_loss += loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num += 1
    loss = total_loss / iter_num
    evaluator=get_acc_evaluator()
    acc = evaluator(torch.cat(y_pred, dim=0), torch.cat(y_true))
    return loss, acc, model

def batch_test(args, model, feats, labels, loader):
    preds, true= [], []
    
    for batch in loader:
        batch_feats = feats[batch].to(args.device)
        true.append(labels[batch].to(args.device))
        with torch.no_grad():
            model.eval()
            output_att=model(batch_feats) 
        preds.append(torch.argmax(output_att, dim=-1))
    true = torch.cat(true)
    preds = torch.cat(preds, dim=0)
    
    evaluator=get_acc_evaluator()
    res = evaluator(preds, true)
    return res


if __name__ == "__main__":
    args=pars()
    args=args_setting(args)

    g, labels, train_nid, val_nid, test_nid, unlabeled_nid, args = load_dataset(args)
    feats_test, labels_test = data_transductive(args, g, labels, train_nid, val_nid, test_nid, unlabeled_nid)    
    feats_train, labels_train, train_nid = data_inductive(args, g, labels, val_nid, test_nid)

    train_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid), len(train_nid)+len(val_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(torch.arange(len(train_nid)+len(val_nid), len(train_nid)+len(val_nid)+len(test_nid)), batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    
    with open(args.initial_model_result,'a') as f:
        f.writelines("Train initial model")
        f.write('\n') 

    for num_hops in np.arange(1,args.num_max_hops+1):
        args.num_hops = num_hops
        set_seed(args.seed)
        model = sgc(args.in_size,  args.n_classes, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
        loss_fcn =nn.CrossEntropyLoss() 
        optimizer =torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_training(args, model, loss_fcn, optimizer, feats_train, labels_train, feats_test, labels_test, train_loader, val_loader, test_loader)

           
    