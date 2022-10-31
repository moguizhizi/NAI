import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset='flickr'
hyper_settings='./results/'+dataset+'/searched_result/hyper_settings.npy'
t=0.4989 # val acc threshold

# [Ts, Tmin, dif_dim_ratio, Tmax, out_num_list, 
# val_acc_ori, val_acc_p, 
# [flops_batch_ori, flops_batch_pro, speed_flops_x_batch, speed_flops_p_batch],
# testset_acc])  

fig_file='./results/'+dataset+'/searched_result/Validation_Flops.png'
result='./results/'+dataset+'/searched_result/result.txt'

buf=[]
FLOPs_speedup=[]
val_acc=[]
speed=[]

with open(hyper_settings, 'rb') as f:
    file = pickle.load(f)
for data in file:
    val_acc.append(data[6])
    FLOPs_speedup.append(data[7][3])
    if round(data[6],4)>=t:
        buf.append(data)

for i in buf:
    speed.append(i[7][3])
loc=np.argmax(speed)
with open(result,'a') as f:
    f.write(str(buf[loc]) )
    f.write('\n') 

fig = plt.figure(figsize=(6,4)) 
plt.scatter(val_acc,FLOPs_speedup)
plt.ylabel('Reduced FLOPs (%)')
plt.xlabel('Val_acc score')
plt.grid()
plt.tight_layout()
plt.savefig(fig_file)

