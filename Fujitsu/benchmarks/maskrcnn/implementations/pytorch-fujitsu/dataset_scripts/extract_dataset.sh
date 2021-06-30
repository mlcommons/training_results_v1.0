mkdir $DATASET_DIR/coco2017
mv train2017.zip $DATASET_DIR/coco2017
mv val2017.zip $DATASET_DIR/coco2017
mv annotations_trainval2017.zip $DATASET_DIR/coco2017

cd $DATASET_DIR/coco2017
dtrx --one=here annotations_trainval2017.zip
dtrx train2017.zip
dtrx val2017.zip
