mkdir raw-data-dir
cd raw-data-dir
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging

cd ../..
python preprocess_dataset.py --data_dir raw-data-dir/kits19/data --results_dir /data
