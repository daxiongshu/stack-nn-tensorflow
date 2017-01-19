def write_list(names,columns):
    for i in range(3):
	for j in ['train','test']:
	    fo = open('nn/data/%s%d.list'%(j,i),'w')
	    fo.write('name,columns\n')
	    tag = 'tr' if j=='train' else 'va'
	    for name,col in zip(names,columns):
		fo.write('"%s.%s.%d","%s"\n'%(name,tag,i,col))
	    fo.close()

names = ["../cvdata/train_meta.csv",

	"../cvdata/cv_0.690716",
	"../cvdata/cv_0.690598",
	"../cvdata/ffm2_valid_k16_eta0.050",
	"../cvdata/mt_cv_0.681601",
	"../cvdata/ftrl_va_group",

	"../cvdata/2way_try1",
	"../cvdata/2way_try2",

	"../cvdata/cv_0.692248",
	"../cvdata/cv_0.692143",
	#"../cvdata/lat1",
	"../cvdata/cv_0.694441",
	"../cvdata/takuya1",
	#"../cvdata/takuya_features",
	"../cvdata/takuya2",
	"../cvdata/takuya3",
	"../cvdata/takuya4",	
#	"../cvdata/fm_0.693821_cv_13",
#	"../cvdata/fm_0.693821_cv_14",
#	"../cvdata/fm_0.693821_cv_15",
#       "../cvdata/fm_0.693821_cv_16",
#	"../cvdata/fm_0.693821_cv_17",
#        "../cvdata/fm_0.693821_cv_18",
#        "../cvdata/fm_0.693821_cv_19",
#        "../cvdata/fm_0.693821_cv_20",
#	"../cvdata/fm_0.693821_cv_21",
#	"../cvdata/fm_0.693821_cv_22",
#        "../cvdata/fm_0.693821_cv_23",
	#"../cvdata/fm_0.693821_cv_24",
	]
cols = ["sffm0,sffm1,sffm2,sffm3,sffm4,sffm5,sffm6,sffm7,sffm8,sffm9,ffm0",
	"clicked","clicked","clicked","clicked","clicked",
	"neighbor_ad_document_id,neighbor_ad_leak,neighbor_ad_doc_after_click,ad_id_document_id,ad_id_leak,ad_id_doc_after_click,document_idx_document_id,document_idx_leak,document_idx_doc_after_click",
	"ad_id_category_id,ad_id_entity_id,ad_id_source_id,ad_id_publisher_id,campaign_id_category_id,campaign_id_entity_id,campaign_id_source_id,campaign_id_publisher_id,advertiser_id_category_id,advertiser_id_entity_id,advertiser_id_source_id,advertiser_id_publisher_id",
	"clicked",
	"clicked",
	#"lat0",
	"clicked",
        "clicked",
	#"doc_dot_doc,doc_dot_doc1,doc_dot_doc_categories_topics,doc_dot_doc_categories_entities,doc_dot_doc_topics_entities,doc_dot_doc_topics_entities_entities,doc_dot_doc_source_id,doc_dot_doc_publisher_id,doc_dot_doc_topics,doc_dot_doc_categories,doc_dot_doc_entities,user_dot_doc,user_dot_doc1,user_dot_doc_categories_topics,user_dot_doc_categories_entities,user_dot_doc_topics_entities,user_dot_doc_topics_entities_entities,user_dot_doc_source_id,user_dot_doc_publisher_id,user_dot_doc_topics,user_dot_doc_categories,user_dot_doc_entities,norm_user_dot_doc,norm_user_dot_doc1,norm_user_dot_doc_categories_topics,norm_user_dot_doc_categories_entities,norm_user_dot_doc_topics_entities,norm_user_dot_doc_topics_entities_entities,norm_user_dot_doc_source_id,norm_user_dot_doc_publisher_id,norm_user_dot_doc_topics,norm_user_dot_doc_categories,norm_user_dot_doc_entities",
	"clicked",
        "clicked",
	"clicked",
	#"document_id-ad_id,document_id-document_idx,document_id-campaign_id,document_id-advertiser_id,document_id-entity_idx,document_id-source_idx,document_id-publisher_idx,document_id-category_idx,document_id-topic_idx,document_id-source_id_leak,document_id-publisher_id_leak,document_id-leak",
        #"platform-ad_id,platform-document_idx,platform-campaign_id,platform-advertiser_id,platform-entity_idx,platform-source_idx,platform-publisher_idx,platform-category_idx,platform-topic_idx,platform-source_id_leak,platform-publisher_id_leak,platform-leak",
        #"geo_location-ad_id,geo_location-document_idx,geo_location-campaign_id,geo_location-advertiser_id,geo_location-entity_idx,geo_location-source_idx,geo_location-publisher_idx,geo_location-category_idx,geo_location-topic_idx,geo_location-source_id_leak,geo_location-publisher_id_leak,geo_location-leak",
        #"entity_id-ad_id,entity_id-document_idx,entity_id-campaign_id,entity_id-advertiser_id,entity_id-entity_idx,entity_id-source_idx,entity_id-publisher_idx,entity_id-category_idx,entity_id-topic_idx,entity_id-source_id_leak,entity_id-publisher_id_leak,entity_id-leak",
	#"source_id-ad_id,source_id-document_idx,source_id-campaign_id,source_id-advertiser_id,source_id-entity_idx,source_id-source_idx,source_id-publisher_idx,source_id-category_idx,source_id-topic_idx,source_id-source_id_leak,source_id-publisher_id_leak,source_id-leak",
        #"publisher_id-ad_id,publisher_id-document_idx,publisher_id-campaign_id,publisher_id-advertiser_id,publisher_id-entity_idx,publisher_id-source_idx,publisher_id-publisher_idx,publisher_id-category_idx,publisher_id-topic_idx,publisher_id-source_id_leak,publisher_id-publisher_id_leak,publisher_id-leak",
        #"category_id-ad_id,category_id-document_idx,category_id-campaign_id,category_id-advertiser_id,category_id-entity_idx,category_id-source_idx,category_id-publisher_idx,category_id-category_idx,category_id-topic_idx,category_id-source_id_leak,category_id-publisher_id_leak,category_id-leak",
        #"topic_id-ad_id,topic_id-document_idx,topic_id-campaign_id,topic_id-advertiser_id,topic_id-entity_idx,topic_id-source_idx,topic_id-publisher_idx,topic_id-category_idx,topic_id-topic_idx,topic_id-source_id_leak,topic_id-publisher_id_leak,topic_id-leak",	
	#"day-ad_id,day-document_idx,day-campaign_id,day-advertiser_id,day-entity_idx,day-source_idx,day-publisher_idx,day-category_idx,day-topic_idx,day-source_id_leak,day-publisher_id_leak,day-leak",
	#"hour-ad_id,hour-document_idx,hour-campaign_id,hour-advertiser_id,hour-entity_idx,hour-source_idx,hour-publisher_idx,hour-category_idx,hour-topic_idx,hour-source_id_leak,hour-publisher_id_leak,hour-leak",
	#"weekday-ad_id,weekday-document_idx,weekday-campaign_id,weekday-advertiser_id,weekday-entity_idx,weekday-source_idx,weekday-publisher_idx,weekday-category_idx,weekday-topic_idx,weekday-source_id_leak,weekday-publisher_id_leak,weekday-leak",
	#"doc_after_click-ad_id,doc_after_click-document_idx,doc_after_click-campaign_id,doc_after_click-advertiser_id,doc_after_click-entity_idx,doc_after_click-source_idx,doc_after_click-publisher_idx,doc_after_click-category_idx,doc_after_click-topic_idx,doc_after_click-source_id_leak,doc_after_click-publisher_id_leak,doc_after_click-leak",
	]

write_list(names,cols)

