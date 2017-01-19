def write_list(names,columns):
    if True:
	for c,j in enumerate(['train','test']):
	    fo = open('nn/data/sub_%s.list.base'%(j),'w')
	    fo.write('name,columns\n')
	    for name,col in zip(names[c],columns):
		fo.write('"%s","%s"\n'%(name,col))
	    fo.close()

names1 =[
	"../data/meta/ffm-train-dataWeight4-1__406-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.3_K8_bag1.out" 
	#"../data/meta/cv_0.694441_cv.csv",  
	]
names2 =[ 
	"../data/meta/ffm-train-dataWeight4-1__406-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.3_K8_T12_bag1.out"
	#"../data/meta/cv_0.694441_sub.csv",
	]
names = (names1,names2)
cols = ['clicked']
write_list(names,cols)

