import pickle
import numpy as np
import re

SSD_LAYERS = ['expand_trans_conv', 'expand_conv'] # Layers that are not part of the backbone
BATCHNORM_PARAMS = ['beta', 'gamma', 'running_mean', 'running_var'] 
BATCHNORM2_PARAMS = ['beta2', 'gamma2', 'running_mean2', 'running_var2'] 

def pretrain_backbone(param_dict,
                      picklefile_name,
                      layout='NCHW',
                      backbone_prefix='ssd0_resnetmlperf0_'):
    with open(picklefile_name, 'rb') as picklefile:
        pretrained_dict = pickle.load(picklefile)

    for param_name in param_dict.keys():
        # Skip layers not part of the backbone
        if any(n in param_name for n in SSD_LAYERS):
            continue

        # convert parameter name to match the names in the pretrained file
        pretrained_param_name = param_name
        # Remove backbone_prefix from name
        pretrained_param_name = pretrained_param_name.replace(backbone_prefix, '')
        # 'batchnormaddrelu' uses 'moving' rather than 'running' for mean/var
        pretrained_param_name = pretrained_param_name.replace('moving', 'running')
        # for fused conv2d+bn+relu, massage the name a bit
        if "weight" in pretrained_param_name:
            pretrained_param_name = pretrained_param_name.replace('convbn', 'conv')
        if "weight2" in pretrained_param_name:
            # turn stage2_conv1_weight2 into stage2_conv2_weight
            numbers = re.findall("\d+", pretrained_param_name)
            conv_num = int(numbers[1])+1
            pretrained_param_name = "stage{}_conv{}_weight".format(numbers[0], conv_num)
        for param in BATCHNORM_PARAMS:
            if param in pretrained_param_name:
                pretrained_param_name = pretrained_param_name.replace('convbn', 'batchnorm')
        for param in BATCHNORM2_PARAMS:
            if param in pretrained_param_name:
                # turn stage2_batchnorm1_gamma2 to stage2_batchnorm2_gamma
                numbers = re.findall("\d+", pretrained_param_name)
                batchnorm_num = int(numbers[1])+1
                pretrained_param_name = "stage{}_batchnorm{}_{}".format(numbers[0], batchnorm_num, param[:-1])

        assert pretrained_param_name in pretrained_dict, \
               f'Can\'t find parameter {pretrained_param_name} in the picklefile'
        param_type = type(pretrained_dict[pretrained_param_name])
        assert isinstance(pretrained_dict[pretrained_param_name], np.ndarray), \
               f'Parameter {pretrained_param_name} in the picklefile has a wrong type ({param_type})'

        pretrained_weights = pretrained_dict[pretrained_param_name]

        if layout == 'NHWC' and pretrained_weights.ndim==4:
            # Place channels into last dim
            pretrained_weights = pretrained_weights.transpose((0, 2, 3, 1))

            # this special case is intended only for the first
            # layer, where the channel count needs to be padded
            # from 3 to 4 for NHWC
            if (pretrained_weights.shape[3]+1)==param_dict[param_name].shape[3]:
                pretrained_weights = np.pad(pretrained_weights,
                                            ((0, 0), (0, 0), (0, 0), (0, 1)),
                                            mode='constant')

        assert param_dict[param_name].shape == pretrained_weights.shape, \
               'Network parameter {} and pretrained parameter {} have different shapes ({} vs {})' \
               .format(param_name, pretrained_param_name, param_dict[param_name].shape, pretrained_weights.shape)
        param_dict[param_name].set_data(pretrained_weights)
