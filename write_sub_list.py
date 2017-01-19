def write_list(names,columns):
    if True:
	for c,j in enumerate(['train','test']):
	    fo = open('nn/data/sub_%s.list'%(j),'w')
	    fo.write('name,columns\n')
	    for name,col in zip(names[c],columns):
		fo.write('"%s","%s"\n'%(name,col))
	    fo.close()

names1 = ["../../stack/data/cv_0.691775_lb_0.69167/train_meta.csv",

	"../data/meta/cv_0.690598_cv.csv",  
	"../data/meta/cv_0.690716_cv.csv", 
	"../data/meta/cv_0.692248_cv.csv",
	"../data/meta/cv_0.692143_cv.csv",
	"../data/meta/mt_cv_0.681601_cv.csv",
	"../data/meta/ftrl_va_group.csv",
	"../data/meta/ffm2_valid_k16_eta0.050.csv",
	#"../data/meta/leak_meta_cv.csv",
	'../data/meta/ffm-train-dataWeight4-1__406-nextView_Wleak_R0.3_K8_bag1.out',
	'../data/meta/ffm-train-dataWeight4-1__407-1.5-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.4_K8_bag1.out',
	'../data/meta/va_xgb.csv',
	'../data/meta/cv_0.694441_cv.csv',
	"../data/meta/ffm-train-dataWeight4-1__406-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.3_K8_bag1.out",	
	"../data/meta/full_try_1va.fm_2way.target",
	"../data/meta/full_try_2va.fm_2way.target",
	]
names2 = ["../../stack/data/cv_0.691775_lb_0.69167/test_meta.csv",

        "../data/meta/cv_0.690598_sub.csv",  
        "../data/meta/cv_0.690716_sub.csv",
	"../data/meta/cv_0.692248_sub.csv",
	"../data/meta/cv_0.692143_sub.csv",
        "../data/meta/mt_cv_0.681601_sub.csv",
	"../data/meta/ftrl_test_group.csv",
	"../data/meta/ffm2_pred_k16_eta0.050.csv",
	#"../data/meta/leak_meta_sub.csv",	
	'../data/meta/ffm-train-dataWeight4-1__406-nextView_Wleak_R0.3_K8_T12_bag1.out',
        '../data/meta/ffm-train-dataWeight4-1__407-1.5-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.4_K8_T23_bag1.out',
        '../data/meta/test_xgb.csv',
        '../data/meta/cv_0.694441_sub.csv',
        "../data/meta/ffm-train-dataWeight4-1__406-nextView-nextViewMulti-nextViewDot-nextViewMultiDot_Wleak_R0.3_K8_T12_bag1.out", 
	"../data/meta/full_try_1test.fm_2way.target",
	"../data/meta/full_try_2test.fm_2way.target",
	]

names = (names1,names2)
cols = ["sffm0,sffm1,sffm2,sffm3,sffm4,sffm5,sffm6,sffm7,sffm8,sffm9,ffm0",
	"clicked","clicked","clicked","clicked","clicked","clicked","clicked",
	#"source_id_leak,publisher_id_leak",
	"clicked","clicked","clicked","clicked","clicked",	
	"neighbor_ad_document_id,neighbor_ad_leak,neighbor_ad_doc_after_click,ad_id_document_id,ad_id_leak,ad_id_doc_after_click,document_idx_document_id,document_idx_leak,document_idx_doc_after_click",
	"ad_id_category_id,ad_id_entity_id,ad_id_source_id,ad_id_publisher_id,campaign_id_category_id,campaign_id_entity_id,campaign_id_source_id,campaign_id_publisher_id,advertiser_id_category_id,advertiser_id_entity_id,advertiser_id_source_id,advertiser_id_publisher_id",
	]

write_list(names,cols)

