import csv

def split(inputname, display, outputdir,name=None):
    if name is None:
    	name = inputname.split('/')[-1]
    
    ftr = [open('%s/%s.tr.%d'%(outputdir,name,i),'w') for i in range(3)]
    fva = [open('%s/%s.va.%d'%(outputdir,name,i),'w') for i in range(3)]
    

    with open(inputname, 'r') as f:
	head = f.readline()
	for i in range(3):
	    ftr[i].write(head)
	    fva[i].write(head)
	with open(display, 'r') as fd:
	    for c,row in enumerate(csv.DictReader(fd)):
		line = f.readline()
		for i in range(3):
		    if row['fold%d'%(i+1)]=='0':
			ftr[i].write(line)
		    elif row['fold%d'%(i+1)]=='1':
			fva[i].write(line)
		if c%1000000 == 0:
		    print c
    for i in range(3):
	ftr[i].close()
	fva[i].close()

def split_display(display, outputdir):
    name = display.split('/')[-1]
    ftr = [open('%s/%s.display.tr.%d'%(outputdir,name,i),'w') for i in range(3)]
    fva = [open('%s/%s.display.va.%d'%(outputdir,name,i),'w') for i in range(3)]
    head = 'display_id,clicked'
    for i in range(3):
	ftr[i].write(head)
	fva[i].write(head)

    with open(display, 'r') as fd:
        for c,row in enumerate(csv.DictReader(fd)):
	    line='%s,%s'%(row['display_id',row.get('clicked','0')])
            for i in range(3):
                if row['fold%d'%(i+1)]=='0':
                    ftr[i].write(line)
                elif row['fold%d'%(i+1)]=='1':
                    fva[i].write(line)
            if c%1000000 == 0:
                print c
    for i in range(3):
        ftr[i].close()
        fva[i].close()



if __name__ == '__main__':
    #split(inputname='../stack/data/cv_0.691775_lb_0.69167/train_meta.csv', display='data/stack_split2.csv', outputdir='cvdata')
    #split_display(display='data/stack_split2.csv', outputdir='cvdata')				
    #split(inputname='../better_split/data/clicks_va.csv', display='data/stack_split2.csv', outputdir='cvdata')
    #split(inputname='../better_split/good/cv_0.691873_lb_0.69122/cv_0.691873/cv.csv', display='data/stack_split2.csv', outputdir='cvdata',name='cv_0.691873')	
    #split(inputname='../better_split/good/cv_0.690716/cv_0.690716/cv.csv', display='data/stack_split2.csv', outputdir='cvdata',name='cv_0.690716')
    #split(inputname='../better_split/good/cv_0.690598/cv_0.690598/cv.csv', display='data/stack_split2.csv', outputdir='cvdata',name='cv_0.690598')
    #split(inputname='data/ffm2_valid_k16_eta0.050.csv', display='data/stack_split2.csv', outputdir='cvdata',name='ffm2_valid_k16_eta0.050')
    split(inputname='data/ftrl_va_group.csv', display='data/stack_split2.csv', outputdir='cvdata',name='ftrl_va_group') 
    split(inputname='data/mt_cv_0.681601/cv.csv', display='data/stack_split2.csv', outputdir='cvdata',name='mt_cv_0.681601')	
