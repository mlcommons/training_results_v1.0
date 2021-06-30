mkdir $DATASET_DIR/coco2017
mv train2017.zip $DATASET_DIR/coco2017
mv val2017.zip $DATASET_DIR/coco2017
mv annotations_trainval2017.zip $DATASET_DIR/coco2017

cd $DATASET_DIR/coco2017
unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
