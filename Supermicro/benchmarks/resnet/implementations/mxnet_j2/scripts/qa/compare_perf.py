import pandas as pd
import numpy as np
import argparse

def getIds(row):
    name = row['LayerName']
    if 'stage' in name :
        n = name.split("_")
        stage = n[0]
        unit = n[1]
        branch = n[2]
        stage = "res"+str(int(stage.replace('stage',''))+1)
        unit = {'1':'a', '2':'b', '3':'c', '4':'d', '5':'e', '6':'f'}[unit.replace('unit','')]
        if 'sc' in name:
            branch = "branch1"
        else:    
            branch = branch.replace("bn","").replace('conv',"")
            branch = {'1':'branch2a','2':'branch2b','3':'branch2c'}[branch]
        return stage, unit, branch 
    return "","",""

def getTag(row, net):
    if row['LayerName'] == 'conv0':
        lname = 'conv1'
    else:
        lname = row['UnitId'] + row['Unit'] + "_" + row['BranchId']
    tmp = net.loc[net['Name']==lname].to_records()
    if tmp:
        return tmp[0][16]
    return 'None'

def print_dict(in_df, key1, key2, ref_name, speedup_filter = None):
    in_df = in_df.dropna()
    in_dict = in_df.to_dict()
    key_list = []
    for k in in_dict:
        key_list.append(k)
    print("-"*124)
    key1_s = key1
    if key1 == None:
        key1_s = ""
    key2_s = key2
    if key2 == None:
        key2_s = ""
    print ("|{:<20}| |{:<20}| |{:<30}| |{:<30}| |{:<10}|".format(key1_s, key2_s,key_list[0][1][-30:],key_list[1][1][-30:],"SpeedUp"))
    print("-"*124)
    data = []
    for sub_key in in_dict[key_list[0]]:
        if(len(sub_key)) > 2:
            val =[sub_key, sub_key] + [ in_dict[key][sub_key] for key in key_list]
        else:
            val =[sub_key[0], sub_key[1]] + [ in_dict[key][sub_key] for key in key_list]
        val_a = "{:4.4f}".format(val[2])
        val_b = "{:4.4f}".format(val[3])
        ref_name = ref_name.replace(".xlsx","")
        if (key_list[0][1] == ref_name) :
            speedup_f = val[2] / val[3]
        elif(key_list[1][1] == ref_name):
            speedup_f = val[3] / val[2]
        else:
            print(key_list[0][1], key_list[1][1], ref_name)
            print(key_list)
            assert False
        speedup = "{:4.4f}x".format(speedup_f)
        if speedup_filter:
            if (speedup_f > speedup_filter or speedup_f < 1.0/speedup_filter):
                out_s = "|{:<20}| |{:<20}| |{:<30}| |{:<30}| |{:<10}|".format(val[0][:20],val[1],val_a, val_b, speedup)
                data.append((speedup, out_s))
        else:
            out_s = "|{:<20}| |{:<20}| |{:<30}| |{:<30}| |{:<10}|".format(val[0][:20],val[1],val_a, val_b, speedup)
            data.append((speedup, out_s))
    data.sort()
    for i, j in data:
        print(j)
    print("-"*124)
    return data


def getConvParam(row):
    tag = row['Tag']
    if "CONV" in tag:
        tag = tag.split("CONV")[1].split('_NHWC')[0].split('X')
        return tag 
    print(tag)

def getKernel(row, df):
    name = row['LayerName']
    ph = row['Phase']
    tmp = df[(df['LayerName'] == name) & (df['Phase'] == ph)]['Kernel']
    return tmp.to_list()[-1].replace("void nhwc_batch_norm_","").replace("nhwc_batch_norm_","")

def getDims(row, df):
    name = row['LayerName']
    ph = row['Phase']
    tmp = df[(df['LayerName'] == name) & (df['Phase'] == ph)]['TensorShapes']
    return tmp.to_list()[-1]

def get_cudnn_commands(mdf, filter_list, ref_df, res_df) :
    flt_list = []
    for i, j in filter_list:
        j = j.replace(" ","").split('||')
        name, phase, speedup = j[0].replace("|",""),j[1],float(j[4].replace("x|",""))
        if speedup < 0.75 :
            flt_list.append((name,phase))
    print(flt_list)
    net = pd.read_csv("qa/RN50Network.csv")
    mdf = mdf[[x in flt_list for x in list(zip(mdf.LayerName,mdf.Phase))]]
    if mdf.empty:
        return
    bdf = mdf[mdf['LayerType']=='BatchNorm']
    if not bdf.empty:
        bdf['Kold'] = bdf.apply(lambda x : getKernel(x, ref_df), axis = 1)
        bdf['Knew'] = bdf.apply(lambda x : getKernel(x, res_df), axis = 1)
        bdf['Dims'] = bdf.apply(lambda x : getDims(x, res_df), axis = 1)
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        pd.set_option('display.max_colwidth',40)
        print(bdf[['LayerName','Kold','Knew','Dims','Phase']])
    
    df = mdf[ (mdf['LayerType']=='Convolution')]
    if not df.empty:
        df['UnitId'] = df.apply(lambda x : getIds(x)[0], axis = 1) 
        df['Unit'] = df.apply(lambda x : getIds(x)[1], axis = 1) 
        df['BranchId'] = df.apply(lambda x : getIds(x)[2], axis = 1)
        df['Tag'] = df.apply(lambda x : getTag(x, net), axis = 1)
        df['N'] = df.apply(lambda x : getConvParam(x)[0], axis = 1)
        df['H'] = df.apply(lambda x : getConvParam(x)[1], axis = 1)
        df['W'] = df.apply(lambda x : getConvParam(x)[2], axis = 1)
        df['C'] = df.apply(lambda x : getConvParam(x)[3], axis = 1)
        df['K'] = df.apply(lambda x : getConvParam(x)[4], axis = 1)
        df['R'] = df.apply(lambda x : getConvParam(x)[5], axis = 1)
        df['S'] = df.apply(lambda x : getConvParam(x)[6], axis = 1)
        df['U'] = df.apply(lambda x : getConvParam(x)[7], axis = 1)
        df['V'] = df.apply(lambda x : getConvParam(x)[8], axis = 1)
        df['padh'] = df.apply(lambda x : getConvParam(x)[9], axis = 1)
        df['padw'] = df.apply(lambda x : getConvParam(x)[10], axis = 1)
        df = df[['LayerName','Phase','GPUDuration(ms)','N','H','W','C','K','R','S','U','V','padh','padw']]
        print(df)
    return df

parser = argparse.ArgumentParser(description='Compare plot_nsight.py dfs')
parser.add_argument('--network', required=True )
parser.add_argument('--batch', required=True )
parser.add_argument('--ifile', required=True )
parser.add_argument('--tol', default=1.002)
parser.add_argument('--filter_type')
parser.add_argument('--filter_phase')
parser.add_argument('--gen_cudnn')
parser.add_argument('--mxnet', default='21.03')
args = parser.parse_args()
args.batch = int(args.batch)
tol = float(args.tol)

ref_name = '{}_b{}_mxnet{}.xlsx'.format(args.network, args.batch, args.mxnet)
ref_path = 'qa/{}'.format(ref_name)

ref_df = pd.read_excel(ref_path)
res_df = pd.read_excel('{}'.format(args.ifile))

df = ref_df.append(res_df, ignore_index=True)
if args.filter_type:
    tfilter = args.filter_type.split(',')
    df = df[df.LayerType.isin(tfilter)]
if args.filter_phase:
    pfilter = args.filter_phase.split(',')
    df = df[df.Phase.isin(tfilter)]
print(df)
pvt_layer_type = pd.pivot_table(df, values=['GPUDuration(ms)'], index=['LayerType','Phase'], columns=['ExperTag'],aggfunc=np.sum)
pvt_layer_name = pd.pivot_table(df, values=['GPUDuration(ms)'], index=['LayerName','Phase'], columns=['ExperTag'],aggfunc=np.sum)
pvt_layer = pd.pivot_table(df, values=['GPUDuration(ms)'], index=['Phase'], columns=['ExperTag'], aggfunc=np.sum)
summary = pd.pivot_table(df, values=['GPUDuration(ms)'], index=['ExperTag'], aggfunc=np.sum)
print(summary)
print("Total Variations")
su = summary.to_records()
print("\t{} = {:.3f} {} = {:.3f} var = {:.2f}x".format(su[0][0], su[0][1], su[1][0], su[1][1], su[1][1] / su[0][1]))
print_dict(pvt_layer, 'Phase', None, ref_name)
print_dict(pvt_layer_type, 'LayerType', 'Phase', ref_name, tol)
layer_speedup = print_dict(pvt_layer_name, 'LayerName', 'Phase', ref_name, tol)
#df = get_cudnn_commands(res_df, layer_speedup, ref_df, res_df)#[['LayerName','Phase','GPUDuration(ms)','N','H','W','C','K','R','S','U','V','padh','padw']]

