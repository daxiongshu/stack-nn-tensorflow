import os
import pandas as pd
import numpy as np
for i in range(5):
    cmd = 'python main.py --train data/sub_train.list --display_train ../../better_split/data/clicks_va.csv  --fea_limit 5 --test data/sub_test.list --display_test ../../input/clicks_test.csv --learning_rate 0.0005 --weight_decay 0.001 --batch_size 96 --sub bag/sub%d.csv --num_epochs 3 --acc_period 100  --normalize 1 --base_train data/sub_train.list.base --base_test data/sub_test.list.base --model pgs_wb'%i
    os.system(cmd)
s = []
for i in range(5):
    s.append(pd.read_csv('bag/sub%d.csv'%i))
s = pd.concat(s, axis=1).values
s = np.mean(s,axis=1)
np.savetxt("bag/sub_ave.csv",s,header='clicked')
 
