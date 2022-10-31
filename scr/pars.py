from module import *

def pars():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--data_dir", type=str, default='../data/')    
    parser.add_argument("--seed", type=int, default=0)

    # model 
    parser.add_argument("--num_max_hops", type=int, default=7)
    parser.add_argument("--hidden_channels", type=float, default=16)
    parser.add_argument("--num_layers", type=float, default=3)

    # optimization
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=30000)
    parser.add_argument("--eval_batch_size", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--test_node_num", type=float, default=500)

    # hyper-para training 
    parser.add_argument("--temperature_off", type=float, default=1.2)
    parser.add_argument("--lambda_off", type=float, default=0.6)
    parser.add_argument("--temperature_on", type=float, default=1.9)
    parser.add_argument("--lambda_on", type=float, default=0.8)
    parser.add_argument("--ensemble_layers", type=float, default=5)

    # hyper-para inference 
    parser.add_argument("--Tmin", type=float, default=0)
    parser.add_argument("--Tmax", type=float, default=2)
    parser.add_argument("--Ts", type=float, default=0.0105)
    parser.add_argument("--Ts_weight", type=float, default=304.6738)
    parser.add_argument("--dif_dim_ratio", type=float, default=0.5)
    # results
    parser.add_argument("--result_fold", type=str, default='searched_result')
    parser.add_argument("--classifier_fold", type=str, default='trained_model')

    args = parser.parse_args()  
    return args

def args_setting(args):

    if args.gpu < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.gpu)
      
    args.classifier_fold='./results/'+args.dataset+'/'+args.classifier_fold+'/'
    if not os.path.exists( args.classifier_fold+'log/'):
        os.makedirs( args.classifier_fold+'log/') 
    if not os.path.exists( args.classifier_fold+'model/'):
        os.makedirs( args.classifier_fold+'model/') 

    args.initial_model_result=args.classifier_fold+'log/'+'initial_model_result.txt'
    args.off_model_result = args.classifier_fold+'log/'+'off_model_result.txt'
    args.on_model_result = args.classifier_fold+'log/'+'on_model_result.txt'

    args.base_model = args.classifier_fold+'model/'+'base_model.pkl'
    args.off_distilled_model=args.classifier_fold+'model/'+'off_distilled_model.pkl'
    args.on_distilled_model=args.classifier_fold+'model/'+'on_distilled_model.pkl'

    args.result_fold='./results/'+args.dataset+'/'+args.result_fold+'/'
    if not os.path.exists(args.result_fold):
        os.makedirs(args.result_fold) 
    args.inference_result=args.result_fold+'/inference_result.txt'
    args.hyper_settings=args.result_fold+'/hyper_settings.npy'

    return args
