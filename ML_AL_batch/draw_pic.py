import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
resultset = pd.read_csv('D:/matlab_project/AL-ML_result/measure/ml_al_os/label_corss/slashdot/reust2_change.csv')
# f1_num = int((len(resultset) + 1) / 2)
f1_num = int(len(resultset) / 3) + 1
# f1_num = int(len(resultset))
result_arr = resultset[['Random','EMAL','LCI','MMU','AUDI','CVIRS','KL_AL','labelpro_jd']]
result_arr = np.array(result_arr)

x = np.linspace(0,f1_num - 1,f1_num)
fig,ax = plt.subplots(figsize = (10,8))

index = [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72]
# index = np.array(index)
random = ax.plot(x,result_arr[index,0],linewidth = 2,marker = '*',markersize = 6)
emal = ax.plot(x,result_arr[index,1],linewidth = 2,marker = 'p',markersize = 6)
lci = ax.plot(x,result_arr[index,2],linewidth = 2,marker = 'd',markersize = 6)
mmu = ax.plot(x,result_arr[index,3],linewidth = 2,marker = 'D',markersize = 6,color='grey')
audi = ax.plot(x,result_arr[index,4],linewidth = 2,marker = '^',markersize = 6,color = 'cadetblue')
cvirs = ax.plot(x,result_arr[index,5],linewidth = 2,marker = 'v',markersize = 6,color = 'orchid')
kl_al = ax.plot(x,result_arr[index,6],linewidth = 2,marker = 's',markersize = 6,color = 'mediumslateblue')
my_al = ax.plot(x,result_arr[index,7],linewidth = 2,marker = 'o',markersize = 6,color = 'crimson')


plt.legend(('Random','EMAL','LCI','MMU','AUDI','CVIRS','KL_AL','PLVI-CE'),fontsize=18,framealpha=0.5,loc='lower right')

# x1 = [0,5,10,15,20,25,30]
x1 = np.linspace(0,f1_num,7)
x1 = np.array(x1)
group_labels = ['0%','10%','20%','30%','40%','50%','60%']
plt.xlim(0,6)

plt.xticks(x1,group_labels)
plt.tick_params(labelsize=16,width=2)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font = {'family': 'Times New Roman','weight' : 'bold','size' : 19}
plt.xlabel('Percent of queries in the initial unlabeled set',font)
plt.ylabel('Macro F1',font)

plt.grid(linestyle='--',linewidth=1.5)
b = plt.gca()#Get the handle of the axis
b.spines['bottom'].set_linewidth(1.5)
b.spines['left'].set_linewidth(1.5)
b.spines['right'].set_linewidth(1.5)
b.spines['top'].set_linewidth(1.5)

plt.ylim(0.12,0.18)


plt.savefig(r'D:\matlab_project\AL-ML_result\measure\ml_al_os\python_code_pic\slashdot_macro.tif',dpi=300,bbox_inches='tight')
plt.show()