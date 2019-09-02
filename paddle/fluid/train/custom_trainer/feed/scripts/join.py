#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
This is an example of network building
"""

from __future__ import print_function, division
import paddle
from paddle import fluid

def sparse_cvm_dim(sparse_info):
    return sparse_info['slot_dim'] * len(sparse_info['slots'])

def inference():
    """Build inference network(without loss and optimizer)

    Returns:
        list<Dict>: sparse_inputs
        and
        list<Variable>: inputs
        and
        list<Variable>: outputs
    """
    sparse_cvm = { "name": "cvm_input", "slot_dim" : 11, "slots": [6048,6002,6145,6202,6201,6121,6738,6119,6146,6120,6147,6122,6123,6118,6142,6143,6008,6148,6151,6127,6144,6094,6083,6952,6739,6150,6109,6003,6099,6149,6129,6203,6153,6152,6128,6106,6251,7082,7515,6951,6949,7080,6066,7507,6186,6007,7514,6125,7506,10001,6006,7023,6085,10000,6098,6250,6110,6124,6090,6082,6067,6101,6004,6191,7075,6948,6157,6126,6188,7077,6070,6111,6087,6103,6107,6194,6156,6005,6247,6814,6158,7122,6058,6189,7058,6059,6115,7079,7081,6833,7024,6108,13342,13345,13412,13343,13350,13346,13409,6009,6011,6012,6013,6014,6015,6019,6023,6024,6027,6029,6031,6050,6060,6068,6069,6089,6095,6105,6112,6130,6131,6132,6134,6161,6162,6163,6166,6182,6183,6185,6190,6212,6213,6231,6233,6234,6236,6238,6239,6240,6241,6242,6243,6244,6245,6354,7002,7005,7008,7010,7012,7013,7015,7016,7017,7018,7019,7020,7045,7046,7048,7049,7052,7054,7056,7064,7066,7076,7078,7083,7084,7085,7086,7087,7088,7089,7090,7099,7100,7101,7102,7103,7104,7105,7109,7124,7126,7136,7142,7143,7144,7145,7146,7147,7148,7150,7151,7152,7153,7154,7155,7156,7157,7047,7050,6253,6254,6255,6256,6257,6259,6260,6261,7170,7185,7186,6751,6755,6757,6759,6760,6763,6764,6765,6766,6767,6768,6769,6770,7502,7503,7504,7505,7510,7511,7512,7513,6806,6807,6808,6809,6810,6811,6812,6813,6815,6816,6817,6819,6823,6828,6831,6840,6845,6875,6879,6881,6888,6889,6947,6950,6956,6957,6959,10006,10008,10009,10010,10011,10016,10017,10018,10019,10020,10021,10022,10023,10024,10029,10030,10031,10032,10033,10034,10035,10036,10037,10038,10039,10040,10041,10042,10044,10045,10046,10051,10052,10053,10054,10055,10056,10057,10060,10066,10069,6820,6821,6822,13333,13334,13335,13336,13337,13338,13339,13340,13341,13351,13352,13353,13359,13361,13362,13363,13366,13367,13368,13369,13370,13371,13375,13376,5700,5702,13400,13401,13402,13403,13404,13406,13407,13408,13410,13417,13418,13419,13420,13422,13425,13427,13428,13429,13430,13431,13433,13434,13436,13437,13326,13330,13331,5717,13442,13451,13452,13455,13456,13457,13458,13459,13460,13461,13462,13463,13464,13465,13466,13467,13468,1104,1106,1107,1108,1109,1110,1111,1112,1113,1114,1115,1116,1117,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,13812,13813,6740,1490,1491]} 

    # TODO: build network here
    cvm_input = fluid.layers.data(name='cvm_input', shape=[sparse_cvm_dim(sparse_cvm)], dtype='float32', stop_gradient=False)
    net = cvm_input
    net = fluid.layers.data_norm(input=net, name="bn6048", epsilon=1e-4,
        param_attr={"batch_size":1e4, "batch_sum_default":0.0, "batch_square":1e4})
    lr_x = 1.0
    init_range = 0.2
    fc_layers_size = [511, 255, 255, 127, 127, 127, 127]
    fc_layers_act = ["relu"] * len(fc_layers_size)
    scales_tmp = [net.shape[1]] + fc_layers_size
    scales = []
    for i in range(len(scales_tmp)):
        scales.append(init_range / (scales_tmp[i] ** 0.5))
    for i in range(len(fc_layers_size)):
        net = fluid.layers.fc(
            input = net,
            size = fc_layers_size[i],
            name = 'fc_' + str(i+1), 
            act = fc_layers_act[i],
            param_attr = \
                fluid.ParamAttr(learning_rate=lr_x, \
                initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])),
            bias_attr = \
                fluid.ParamAttr(learning_rate=lr_x, \
                initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=1.0 * scales[i])))
    ctr_output = fluid.layers.fc(net, 1, act='sigmoid', name='ctr')
    
    accessors = [
        { "class": "AbacusSparseJoinAccessor", "input": "sparses", "table_id": 0, "need_gradient": False},
        { "class": "DenseInputAccessor", "input": "vars", "table_id": 1, "need_gradient": True, "async_pull": True},
        { "class": "DenseInputAccessor", "input": "sums", "table_id": 2, "need_gradient": True, "async_pull": True},
        { "class": "LabelInputAccessor", "input": "labels"}
    ]
    monitors = [
        { "name": "epoch_auc", "class": "AucMonitor", "target": ctr_output, "compute_interval": 600 },
        { "name": "day_auc", "class": "AucMonitor", "target": ctr_output, "compute_interval": 86400 }
    ]
    
    return {
        'accessors': accessors, 
        'monitors': monitors, 
        'sparses': [sparse_cvm], 
        'inputs': [cvm_input], 
        'outputs': [ctr_output]
    }

def loss_function(ctr_output):
    """
    Args:
        *outputs: the second result of inference()

    Returns:
        Variable: loss
        and
        list<Variable>: labels
    """
    # TODO: calc loss here

    label = fluid.layers.data(name='label_ctr', shape=ctr_output.shape, dtype='float32')
    loss = fluid.layers.square_error_cost(input=ctr_output, label=label)
    loss = fluid.layers.mean(loss, name='loss_ctr')

    return loss, [label]
