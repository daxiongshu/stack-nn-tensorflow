import pandas as pd
import numpy as np
import h5py
import os
import csv
import gc
from sklearn.utils import shuffle as sk_shuffle
import random

class DataSet:
    def __init__(self, metalist, display, baselist = '', cache='cache', shuffle=False, limit = 5, normalize = 1, softmax=0,mean=None,std=None,num_big=10):
        self.num_big = num_big
	self.softmax = 0#softmax # perform softmax transformation online
	self.files, self.columns = self._parse(metalist)
	self.baselist = baselist
	if self.baselist != '':
	    self.bfiles, self.bcolumns = self._parse(baselist)
	self.normalize = normalize
	self.display = display
	self.cache = cache
	self.limit = limit  # max number of features, above the limit will call online code
	#self.files       # a list of file names: ['xx/m1.csv', 'yy/m2.csv' ...]
        #self.columns     # column names of each file [['clicked'], ['ffm1','ffm2'], ...]

	self.shuffle = shuffle
	self._read()
	self.mean, self.std = mean, std
	if normalize and mean is None:
	    self._normalize()	
	self.reset()

    def _normalize(self):
	print "run normalization ..."
	sum1, sum2 = None, None
	num = 0
	for i in range(self.num_big):
	    self._get_big_batch(batch_id = i)
	    num += self.big_X.shape[0]
	    if i==0:
		sum1 = np.sum(self.big_X,axis=0)
		sum2 = np.sum(self.big_X*self.big_X,axis=0)
	    else:
		sum1 += np.sum(self.big_X,axis=0)
		sum2 += np.sum(self.big_X*self.big_X,axis=0)
	    del self.big_X
            gc.collect()	
	mean1,mean2 = sum1/num, sum2/num
	std = np.sqrt(mean2-mean1*mean1)
	self.mean = mean1
        self.std = std
	print self.mean.shape, self.std.shape	

    def reset(self):
	self.current_big_batch = 0 # 0~9
        self.current_mini_batch = 0
        self.big_X = None
        self.big_Y = None
        self.big_order = range(self.num_big)
	if self.shuffle:
	    random.shuffle(self.big_order)
	
    def _load_row_array(self):
	name = self.display.split('/')[-1]
	cache = self.cache
	cname = '%s/%s.row_array.bin'%(cache, name)
	h5f=h5py.File(cname,'r')
        self.row_array=h5f['dataset_1'][:]
        h5f.close()

    def next_batch(self, batch_size):
	shuffle = self.shuffle 
	if self.big_X is None:
	    self._get_big_batch(batch_id = self.big_order[self.current_big_batch])
	    self.tmp_list = range(len(self.group_idx)-1)
	    self._base_get_big_batch(batch_id = self.big_order[self.current_big_batch])
	    if shuffle:
		random.shuffle(self.tmp_list)
	#print self.row_array
	#print self.group_idx, self.tmp_list
	mini_batch_id = self.current_mini_batch
  	next_batch = mini_batch_id + batch_size
	next_batch = min(next_batch, len(self.tmp_list))
	#print self.current_mini_batch, next_batch
	X, y, G, R = self._extract_batch(self.current_mini_batch, next_batch)	
	done = False
	if self.baselist != '':
	    Xb = self._base_extract_batch(self.current_mini_batch, next_batch)
	if next_batch == len(self.tmp_list):
	    del self.big_X
	    gc.collect()
	    self.big_X = None
	    self.big_base_X = None
	    self.current_big_batch += 1
	    if self.current_big_batch==self.num_big:
		self.reset()
		done = True
	    self.current_mini_batch = 0
	else:
	    self.current_mini_batch = next_batch 
	#X = (X - np.mean(X,0))/np.std(X,0)
	if self.baselist == '':
	    return X, y, G, R, done
	else:
	    return X, y, G, R, done, Xb

    def _softmax(self, x):
	# perform softmax on the 2nd dimenstion
	for i in range(x.shape[1]):
	    #tmpmax = np.max(x[:,i])
	    tmp = np.exp(x[:,i])
	    sumtmp = np.sum(tmp)
	    x[:,i] = tmp/sumtmp
	return x

    def _base_extract_batch(self, start, end):
        X, R = [], [0]
        for i in range(start,end):
            idx = self.tmp_list[i]
            #idx = self.group_idx[idx]
            s_, e_ = self.group_idx[idx], self.group_idx[idx+1]
            R.append(e_-s_+R[-1])
            #print s_, e_
            if self.softmax:
                X.append(self._softmax(self.big_base_X[s_:e_,:]))
            else:
                X.append(self.big_base_X[s_:e_,:])
	X = np.vstack(X)
	return X

    def _extract_batch(self, start, end):
	X, y, G, R = [],[],[],[0]
	for i in range(start,end):
	    idx = self.tmp_list[i]
	    #idx = self.group_idx[idx]
	    s_, e_ = self.group_idx[idx], self.group_idx[idx+1]
	    R.append(e_-s_+R[-1])
	    #print s_, e_
	    if self.softmax:
		X.append(self._softmax(self.big_X[s_:e_,:]))
	    else:
	        X.append(self.big_X[s_:e_,:])
	    if len(self.big_Y.shape) == 2: # no label, just group
		y.append(self.big_Y[s_:e_,1])	
	        G.append(self.big_Y[s_:e_,0])
	    else:
	 	G.append(self.big_Y[s_:e_])	
        X = np.vstack(X)
	if len(y) == 0:
	    y = np.zeros(X.shape[0])
	else:
	    y = np.concatenate(y)
	G = np.concatenate(G)
	return X, y, G, R		

    def sanity_check(self):
	print (self.row_array)
	print ()
        for i in range(self.num_big):
	    self._get_big_batch(i)
	    print (i)
	    #print self.big_X
	    print (self.big_Y)
	    print (self.group_idx)
	    print ()
    def _base_get_big_batch(self, batch_id):
	if self.baselist=='':
	    self.big_base_X = None
	    return
        X = []
        cache = self.cache
        for f in self.bfiles:
            name = f.split('/')[-1]
            cname = '%s/%s_%d.bin'%(cache,name,batch_id)
            assert(os.path.exists(cname), "Meta bin data not exist!")
            h5f=h5py.File(cname,'r')
            train=h5f['dataset_1'][:]
            h5f.close()
            if len(train.shape)==1:
                train = np.reshape(train, [train.shape[0], 1])
            X.append(train)
	self.big_base_X = np.hstack(X)
	
    def _get_big_batch(self, batch_id):
	X = []
	cache = self.cache
	for f in self.files:
	    name = f.split('/')[-1]
	    cname = '%s/%s_%d.bin'%(cache,name,batch_id)
	    assert(os.path.exists(cname), "Meta bin data not exist!")
	    h5f=h5py.File(cname,'r')
	    train=h5f['dataset_1'][:]
    	    h5f.close()
	    if len(train.shape)==1:
		train = np.reshape(train, [train.shape[0], 1])	
	    X.append(train)
	    del train
	self.big_X = np.hstack(X)
	if self.normalize and self.mean is not None:
	    #self.big_X = (self.big_X - np.mean(self.big_X,0))/np.std(self.big_X,0)
	    self.big_X = (self.big_X - self.mean)/self.std
	#print "Load big batch", batch_id, self.big_X.shape
	del X
	gc.collect()

	display = self.display
	name = display.split('/')[-1]
        cname = '%s/%s_%d.bin'%(cache, name, batch_id)
	assert(os.path.exists(cname), "Display bin data not exist!")
	h5f=h5py.File(cname,'r')
        self.big_Y=h5f['dataset_1'][:]
        h5f.close()
	
	cname = '%s/%s_%d.group.bin'%(cache,name, batch_id)
	h5f=h5py.File(cname,'r')
        self.group_idx=h5f['dataset_1'][:]
        h5f.close()

	

    def _parse(self, metalist):
	files, columns = [], []
	with open(metalist,'r') as f:
	    for c,row in enumerate(csv.DictReader(f)):
		files.append(row['name'])
		columns.append(row['columns'].split(','))
	return files, columns

    def _read(self):
	cache = self.cache
	display = self.display

	name = display.split('/')[-1]
	cname = '%s/%s_0.bin'%(cache, name)
	if os.path.exists(cname) == False:
	    self._RW_display_to_bin()
	cname = '%s/%s.group'%(cache, name)
	if True:
	    with open(cname,'r') as f:
	    	self.num_groups = int(f.readline().strip())
	    print ("Total number of groups", self.num_groups)
	for f,col in zip(self.files, self.columns):
	    name = f.split('/')[-1]
	    for i in range(self.num_big):
	        cname = '%s/%s_%d.bin'%(cache,name,i)
		#print cname
	        if os.path.exists(cname) == False:
		    if len(col)<self.limit:
			print ("build data in whole-data-in-memory mode", f)
		        self._RW_meta_to_bin(f,col)
		    else:
			print ("build data in online mode", f)
			self._RW_meta_to_bin_online(f,col)
	
	if self.baselist=='':
	    print ("Build Data set done!")
	    return

	for f,col in zip(self.bfiles, self.bcolumns):
            name = f.split('/')[-1]
            for i in range(self.num_big):
                cname = '%s/%s_%d.bin'%(cache,name,i)
                #print cname
                if os.path.exists(cname) == False:
                    if len(col)<self.limit:
                        print ("build data in whole-data-in-memory mode", f)
                        self._RW_meta_to_bin(f,col)
                    else:
                        print ("build data in online mode", f)
                        self._RW_meta_to_bin_online(f,col)

	print ("Build Data set done!")

    def _RW_display_to_bin(self):
	display = self.display
	cache = self.cache
	shuffle = self.shuffle
	num_groups = 0
	name = display.split('/')[-1]
	row_array = []

        if True:
	    cname = '%s/%s.group'%(cache, name)
	    if os.path.exists(cname) == True:
		with open(cname,'r') as f:
		    num_groups = int(f.readline().strip())
	    else:	    
                with open(display, 'r') as f:
		    last = ''
		    for c,row in enumerate(csv.DictReader(f)):
			if last!=row['display_id']:
                    	    num_groups += 1
			    row_array.append(c)
			
			last = row['display_id']
		    row_array.append(c+1)
		assert(len(row_array)==num_groups+1)
		with open(cname,'w') as f:
		    f.write('%d\n'%num_groups)
		print ("Total number of groups", num_groups)
	cname = '%s/%s.row_array.bin'%(cache, name)
	h5f=h5py.File(cname,'w')
        h5f.create_dataset('dataset_1', data=np.array(row_array))
        h5f.close()

	total = []
	dids = []
	groups = []
	for i in range(self.num_big):
	    dids.append([])
	    groups.append([0])
   	if True:
            with open(display, 'r') as d:
		last = ''
                dc = 0
		local_row_count = 0
		for c,row in enumerate(csv.DictReader(d)):
		    if last!=row['display_id']:
	    		dc += 1
			if shuffle :
                            idx = dc%self.num_big
                    	else:
                            step = (num_groups/self.num_big)
                            if step<1:
                            	step = 1
                            idx = int(dc / step)
                            if idx>self.num_big-1:
                            	idx = self.num_big-1
			#groups[idx].append(dc-1)
			if last!='':
			    groups[last_idx].append(local_row_count + groups[last_idx][-1])
			local_row_count = 0
			last_idx = idx
		    last = row['display_id']
		    local_row_count += 1
		    cname = '%s/%s_%d.bin'%(cache,name,idx)

		    if os.path.exists(cname) == False:
			if 'clicked' in row:
		    	    dids[idx].append([int(row['display_id']), int(row['clicked'])])
			else:
			    dids[idx].append(int(row['display_id']))
		groups[last_idx].append(local_row_count + groups[last_idx][-1])
	for i in range(self.num_big):
	    cname = '%s/%s_%d.bin'%(cache,name,i)
	    if os.path.exists(cname) == False:    
	        did = np.array(dids[i])
	        h5f=h5py.File(cname,'w')
    	        h5f.create_dataset('dataset_1', data=did)
    	        h5f.close()
		group = np.array(groups[i])
		cname = '%s/%s_%d.group.bin'%(cache,name,i)
		h5f=h5py.File(cname,'w')
                h5f.create_dataset('dataset_1', data=group)
                h5f.close()

	        print ("read", display, did.shape, i, 'done')	
	        total.append(did.shape[0])
	print ("Total:", sum(total))				    	

    def _RW_meta_to_bin(self, inputname,column):
	display = self.display
        cache = self.cache

	shuffle = self.shuffle
	if not shuffle:
            num_groups = self.num_groups
        name = inputname.split('/')[-1]
	yps = []
        for i in range(self.num_big):
	    yps.append([])

        if True:
            with open(display, 'r') as d:
                with open(inputname, 'r') as f:
                    last = ''
                    dc = 0
                    dreader = csv.DictReader(d)
                    yp = []
                    for c,row in enumerate(csv.DictReader(f)):
                        drow = dreader.next()
                        if last!=drow['display_id']:
                            dc += 1
                        last = drow['display_id']
			if shuffle :
                            idx = dc%self.num_big
                    	else:
			    step = (num_groups/self.num_big)
                            if step<1:
    	                        step = 1
                            idx = int(dc / step)
                            if idx>self.num_big-1:
                            	idx = self.num_big-1
			cname = '%s/%s_%d.bin'%(cache,name,idx)
			if os.path.exists(cname) == False:
                            tmp = [float(row[x]) for x in column]
                            yps[idx].append(tmp)
	total = []
	for i in range(self.num_big):
	    cname = '%s/%s_%d.bin'%(cache,name,i)
	    if os.path.exists(cname) == False:
            	yp = np.array(yps[i])
            	print (inputname, i, yp.shape)
		h5f=h5py.File(cname,'w')
                h5f.create_dataset('dataset_1', data=yp)
                h5f.close()
		total.append(yp.shape[0])
	print ("Total:", sum(total))


    def _RW_meta_to_bin_online(self, inputname, column):
	display = self.display
        cache = self.cache

	shuffle = self.shuffle
	if not shuffle:
	    num_groups = self.num_groups
	name = inputname.split('/')[-1]
	total = []
	for i in range(self.num_big):
	    cname = '%s/%s_%d.bin'%(cache,name,i)
	    if os.path.exists(cname) == True:
		continue
	    with open(display, 'r') as d:
	        with open(inputname, 'r') as f:
		    last = ''
		    dc = 0
		    dreader = csv.DictReader(d)
		    yp = []
	    	    for c,row in enumerate(csv.DictReader(f)):
		    	drow = dreader.next()
		    	if last!=drow['display_id']:
			    dc += 1
			last = drow['display_id']
			if shuffle and dc%self.num_big != i:
			    continue
			if not shuffle:
			    step = (num_groups/self.num_big)
                            if step<1:
                                step = 1

			    if dc < step*i:
				continue
			    elif i < self.num_big-1:
				if dc >= step*(i+1):
				    break
		
			tmp = [float(row[x]) for x in column]	
			yp.append(tmp)
	    yp = np.array(yp)
	    print (inputname, i, yp.shape)
	    h5f=h5py.File(cname,'w')
            h5f.create_dataset('dataset_1', data=yp)
            h5f.close()   
	    total.append(yp.shape[0])
	print ("Total:", sum(total))

def get_num_fea( metalist):
    columns = []
    with open(metalist,'r') as f:
        for c,row in enumerate(csv.DictReader(f)):
            columns.append(len(row['columns'].split(',')))
    return sum(columns)

def write_sub(yp,name):
    s = pd.DataFrame({"clicked":yp})
    s.to_csv(name,index=False)	
	
if __name__ == '__main__':
    ds = DataSet(display = '../../../input/clicks_test.csv', files=[], columns=[], cache='../cache', shuffle=False)
