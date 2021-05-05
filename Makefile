upload_data_to_s3:
	aws s3 sync ./data/models s3://neural-rock/data/models
	aws s3 sync ./data/Images_PhD_Miami/Leg194/img s3://neural-rock/data/Images_PhD_Miami/Leg194/img
	aws s3 sync ./data/Images_PhD_Miami/Leg194/ROI s3://neural-rock/data/Images_PhD_Miami/Leg194/ROI
	aws s3 cp ./data/Data_Sheet_GDrive_new.xls s3://neural-rock/data/Data_Sheet_GDrive_new.xls