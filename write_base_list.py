def write_list(names,columns):
    for i in range(3):
	for j in ['train','test']:
	    fo = open('nn/data/%s%d.list.base'%(j,i),'w')
	    fo.write('name,columns\n')
	    tag = 'tr' if j=='train' else 'va'
	    for name,col in zip(names,columns):
		fo.write('"%s.%s.%d","%s"\n'%(name,tag,i,col))
	    fo.close()

names = [#"../cvdata/train_meta.csv",
	#"../cvdata/cv_0.694441",
	#"../cvdata/cv_0.694441_cv_leak",
	"../cvdata/takuya1"
	#"../cvdata/xgb_cv_0.691432_lb_0.69531"
	]
cols = [#"sffm0,sffm1,sffm2,sffm3,sffm4,sffm5,sffm6,sffm7,sffm8,sffm9,ffm0",
	"clicked",
	]

write_list(names,cols)

