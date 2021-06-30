import math
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.base import numeric_types
from mxnet import symbol

class FusedConv2DBNRelu(HybridBlock):
    def __init__(self, kernel_size, layout, num_filter, input_shape, padding=0, strides=1,
            prefix=None, params=None, weight_initializer=None, scale=True, center=True,
            beta_initializer='zeros', gamma_initializer='ones',
            running_mean_initializer='zeros', running_variance_initializer='ones',
            cudnn_algo_fwd=-1, cudnn_algo_bwd_data=-1, cudnn_algo_bwd_filter=-1,
            cudnn_tensor_core_only=False, cudnn_algo_verbose=False,
            cudnn_algo_fwd_prec='None', cudnn_algo_bwd_prec='None',
            workspace=1024):
        super(FusedConv2DBNRelu, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.layout = layout
            self.num_filter = num_filter
            self.weight_initializer = weight_initializer
            self.beta_initializer = beta_initializer
            self.gamma_initializer = gamma_initializer
            self.running_mean_initializer = running_mean_initializer
            self.running_variance_initializer = running_variance_initializer
            self.scale = scale
            self.center = center
            self.input_shape = input_shape
            self.cudnn_algo_fwd = cudnn_algo_fwd
            self.cudnn_algo_bwd_data = cudnn_algo_bwd_data
            self.cudnn_algo_bwd_filter = cudnn_algo_bwd_filter
            self.cudnn_tensor_core_only = cudnn_tensor_core_only
            self.cudnn_algo_verbose = cudnn_algo_verbose
            self.cudnn_algo_fwd_prec = cudnn_algo_fwd_prec
            self.cudnn_algo_bwd_prec = cudnn_algo_bwd_prec
            self.workspace = workspace
            if isinstance(kernel_size, numeric_types):
                self.kernel_size = (kernel_size,)*2
            else:
                self.kernel_size = kernel_size
            assert len(self.kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"

            if isinstance(strides, numeric_types):
                self.strides = (strides,)*len(self.kernel_size)
            else:
                self.strides = strides
            assert len(self.strides) == 2, "strides must be a number or a list of 2 ints"

            if isinstance(padding, numeric_types):
                self.padding = (padding,)*len(self.kernel_size)
            else:
                self.padding = padding
            assert len(self.padding) == 2, "padding must be a number or a list of 2 ints"
            # only support default dilation
            self.dilation = (1,)*len(self.kernel_size)

            self._kwargs = {
                'kernel': self.kernel_size, 
                'stride': self.strides, 
                'dilate': self.dilation,
                'pad': self.padding, 
                'num_filter': self.num_filter,  ## channels on Conv2D
                'num_group': 1,
                'no_bias': True,
                'layout': self.layout,
                'cudnn_algo_fwd': self.cudnn_algo_fwd,
                'cudnn_algo_bwd_data': self.cudnn_algo_bwd_data,
                'cudnn_algo_bwd_filter': self.cudnn_algo_bwd_filter,
                'cudnn_tensor_core_only': self.cudnn_tensor_core_only,
                'cudnn_algo_verbose': self.cudnn_algo_verbose,
                'cudnn_algo_fwd_prec': self.cudnn_algo_fwd_prec,
                'cudnn_algo_bwd_prec': self.cudnn_algo_bwd_prec,
                'workspace': self.workspace
            }

            #weight shape for Convolution
            data = symbol.var('data', shape=self.input_shape)
            op = getattr(symbol, "Convolution")
            sym = op(data, **self._kwargs)
            self.weight_shape = sym.infer_shape_partial()[0][1]
            self.weight = self.params.get('weight', shape=self.weight_shape,
                    init=self.weight_initializer,
                    allow_deferred_init=True)

            #elem count for BNStatsFinalize
            YH = math.floor((self.input_shape[self.layout.find('H')] +2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.strides[0]) + 1
            YW = math.floor((self.input_shape[self.layout.find('W')] +2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.strides[1]) + 1
            self.elem_count = self.input_shape[self.layout.find('N')] * YH * YW

            # batchnorm parameters
            self.gamma = self.params.get('gamma', grad_req='write' if self.scale else 'null',
                                         shape=(self.num_filter,), init=self.gamma_initializer,
                                         allow_deferred_init=True,
                                         differentiable=self.scale)
            self.beta = self.params.get('beta', grad_req='write' if self.center else 'null',
                                        shape=(self.num_filter,), init=self.beta_initializer,
                                        allow_deferred_init=True,
                                        differentiable=self.center)
            self.running_mean = self.params.get('running_mean', grad_req='null',
                                                shape=(self.num_filter,),
                                                init=self.running_mean_initializer,
                                                allow_deferred_init=True,
                                                differentiable=False)
            self.running_var = self.params.get('running_var', grad_req='null',
                                               shape=(self.num_filter,),
                                               init=self.running_variance_initializer,
                                               allow_deferred_init=True,
                                               differentiable=False)

    def hybrid_forward(self, F, x, weight, gamma, beta, running_mean, running_var):
        #symbolic call
        y, s, sq_s = F.NormConvolution(x, weight=weight, fix_gamma=False, no_norm=True,
                                       num_filter=self.num_filter,
                                       kernel=self.kernel_size, layout=self.layout, stride=self.strides, pad=self.padding)
        (eq_scale, eq_bias, s_mean, s_std,
         gamma_out, beta_out) = F.BNStatsFinalize(s, sq_s, gamma, beta, running_mean,
                                                  running_var, fix_gamma=False,
                                                  output_mean_var=True,
                                                  elem_count=self.elem_count)
        y, _ = F.ScaleBiasAddRelu(dataX=y, x_equiv_scale=eq_scale, x_equiv_bias=eq_bias,
                                  x_gamma=gamma_out, x_beta=beta_out, x_mean=s_mean,
                                  x_invvar=s_std, layout=self.layout, act_type='relu',
                                  dual_scale_bias=False, fused_add=False)
        return y

    def cast(self, dtype):
        if (dtype == 'float16'):
            self.weight.cast('float16')


class FusedConv2DBNAddRelu(FusedConv2DBNRelu):
    def __init__(self, **kwargs):
        super(FusedConv2DBNAddRelu, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, z, weight, gamma, beta, running_mean, running_var):
        #symbolic call
        y, s, sq_s = F.NormConvolution(x, weight=weight, fix_gamma=False, no_norm=True,
                                       num_filter=self.num_filter,
                                       kernel=self.kernel_size, layout=self.layout, stride=self.strides, pad=self.padding)
        (eq_scale, eq_bias, s_mean, s_std,
         gamma_out, beta_out) = F.BNStatsFinalize(s, sq_s, gamma, beta, running_mean,
                                                  running_var, fix_gamma=False,
                                                  output_mean_var=True,
                                                  elem_count=self.elem_count)
        y, _ = F.ScaleBiasAddRelu(dataX=y, dataZ=z, x_equiv_scale=eq_scale, x_equiv_bias=eq_bias,
                                  x_gamma=gamma_out, x_beta=beta_out, x_mean=s_mean,
                                  x_invvar=s_std, layout=self.layout, act_type='relu',
                                  dual_scale_bias=False, fused_add=True)
        return y


class FusedConv2DBNDualAddRelu(FusedConv2DBNRelu):
    def __init__(self, kernel_size2, input2_shape, padding2=0, strides2=1, **kwargs):
        super(FusedConv2DBNDualAddRelu, self).__init__(**kwargs)
        with self.name_scope():
            self.input2_shape = input2_shape
            if isinstance(kernel_size2, numeric_types):
                self.kernel_size2 = (kernel_size2,)*2
            else:
                self.kernel_size2 = kernel_size2
            assert len(self.kernel_size2) == 2, "kernel_size2 must be a number or a list of 2 ints"

            if isinstance(strides2, numeric_types):
                self.strides2 = (strides2,)*len(self.kernel_size)
            else:
                self.strides2 = strides2
            assert len(self.strides2) == 2, "strides2 must be a number or a list of 2 ints"

            if isinstance(padding2, numeric_types):
                self.padding2 = (padding2,)*len(self.kernel_size)
            else:
                self.padding2 = padding2
            assert len(self.padding2) == 2, "padding2 must be a number or a list of 2 ints"
            
            self._kwargs2 = {
                'kernel': self.kernel_size2, 
                'stride': self.strides2, 
                'dilate': self.dilation,
                'pad': self.padding2, 
                'num_filter': self.num_filter,  ## channels on Conv2D
                'num_group': 1,
                'no_bias': True,
                'layout': self.layout,
                'cudnn_algo_fwd': self.cudnn_algo_fwd,
                'cudnn_algo_bwd_data': self.cudnn_algo_bwd_data,
                'cudnn_algo_bwd_filter': self.cudnn_algo_bwd_filter,
                'cudnn_tensor_core_only': self.cudnn_tensor_core_only,
                'cudnn_algo_verbose': self.cudnn_algo_verbose,
                'cudnn_algo_fwd_prec': self.cudnn_algo_fwd_prec,
                'cudnn_algo_bwd_prec': self.cudnn_algo_bwd_prec,
                'workspace': self.workspace
            }

            #weight shape for second Convolution
            data = symbol.var('data', shape=self.input2_shape)
            op = getattr(symbol, "Convolution")
            sym = op(data, **self._kwargs2)
            self.weight2_shape = sym.infer_shape_partial()[0][1]
            self.weight2 = self.params.get('weight2', shape=self.weight2_shape,
                    init=self.weight_initializer,
                    allow_deferred_init=True)

            #elem count for BNStatsFinalize
            YH2 = math.floor((self.input2_shape[self.layout.find('H')] +2*self.padding2[0]-self.dilation[0]*(self.kernel_size2[0]-1)-1)/self.strides2[0]) + 1
            YW2 = math.floor((self.input2_shape[self.layout.find('W')] +2*self.padding2[1]-self.dilation[1]*(self.kernel_size2[1]-1)-1)/self.strides2[1]) + 1
            self.elem_count2 = self.input2_shape[self.layout.find('N')] * YH2 * YW2

            #second batchnorm weights
            self.gamma2 = self.params.get('gamma2', grad_req='write' if self.scale else 'null',
                                         shape=(self.num_filter,), init=self.gamma_initializer,
                                         allow_deferred_init=True,
                                         differentiable=self.scale)
            self.beta2 = self.params.get('beta2', grad_req='write' if self.center else 'null',
                                        shape=(self.num_filter,), init=self.beta_initializer,
                                        allow_deferred_init=True,
                                        differentiable=self.center)
            self.running_mean2 = self.params.get('running_mean2', grad_req='null',
                                                shape=(self.num_filter,),
                                                init=self.running_mean_initializer,
                                                allow_deferred_init=True,
                                                differentiable=False)
            self.running_var2 = self.params.get('running_var2', grad_req='null',
                                               shape=(self.num_filter,),
                                               init=self.running_variance_initializer,
                                               allow_deferred_init=True,
                                               differentiable=False)

    def hybrid_forward(self, F, x, x2, weight, gamma, beta, running_mean, running_var,
                       weight2, gamma2, beta2, running_mean2, running_var2):
        ## first conv + bn
        y, s, sq_s = F.NormConvolution(x, weight=weight, fix_gamma=False, no_norm=True,
                                       num_filter=self.num_filter,
                                       kernel=self.kernel_size, layout=self.layout, stride=self.strides, pad=self.padding)
        (eq_scale, eq_bias, s_mean, s_std,
         gamma_out, beta_out) = F.BNStatsFinalize(s, sq_s, gamma, beta, running_mean,
                                                  running_var, fix_gamma=False,
                                                  output_mean_var=True,
                                                  elem_count=self.elem_count)
        ## second conv + bn
        y2, s2, sq_s2 = F.NormConvolution(x2, weight=weight2, fix_gamma=False, no_norm=True,
                                          num_filter=self.num_filter,
                                       kernel=self.kernel_size2, layout=self.layout, stride=self.strides2, pad=self.padding2)
        (eq_scale2, eq_bias2, s_mean2, s_std2,
         gamma_out2, beta_out2) = F.BNStatsFinalize(s2, sq_s2, gamma2, beta2, running_mean2,
                                                    running_var2, fix_gamma=False,
                                                    output_mean_var=True,
                                                    elem_count=self.elem_count2)
        # final fused add+relu
        y, _ = F.ScaleBiasAddRelu(dataX=y, x_equiv_scale=eq_scale, x_equiv_bias=eq_bias,
                                  x_gamma=gamma_out, x_beta=beta_out, x_mean=s_mean,
                                  x_invvar=s_std, dataZ=y2, z_equiv_scale=eq_scale2,
                                  z_equiv_bias=eq_bias2, z_gamma=gamma_out2, z_beta=beta_out2,
                                  z_mean=s_mean2, z_invvar=s_std2, layout=self.layout,
                                  act_type='relu', dual_scale_bias=True, fused_add=True)
        return y

    def cast(self, dtype):
        super(FusedConv2DBNDualAddRelu, self).cast(dtype)
        if (dtype == 'float16'):
            self.weight2.cast('float16')
