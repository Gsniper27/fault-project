import argparse
import numpy as np
parser = argparse.ArgumentParser(description='transformer inversion')
parser.add_argument('--job', default='train', type=str, help='batch size')
parser.add_argument('--dataMode', default='both', type=str, help='batch size')
parser.add_argument('--gpuIndex', default=0, type=int, help='batch size')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--transN', default=1, type=int, help='batch size')
parser.add_argument('--lr0', default=1e-4, type=float, help='batch size')
parser.add_argument('--resDir', default='train/', type=str, help='batch size')
parser.add_argument('--mn', default='256', type=str, help='batch size')
#parser.add_argument('--job', default='train', type=str, help='batch size')
parser.add_argument('--isAtten', default=False, help='batch size', action='store_true')
parser.add_argument('--alpha', default=0.1, type=float, help='batch size')
parser.add_argument('--dw', default=0.15, type=float, help='batch size')
parser.add_argument('--d0', default=0.05, type=float, help='batch size')
parser.add_argument('--modelType', default='mars_sam', type=str, help='batch size')
parser.add_argument('--stepPerMiniBatch', default=8, type=int, help='batch size')
parser.add_argument('--dropout', default=0, type=float, help='batch size')
parser.add_argument('--threshold', default=0.5, type=float, help='batch size')
parser.add_argument('--dataset', default='key.lst', type=str, help='batch size')
parser.add_argument('--datasetAdd', default='', type=str, help='batch size')
parser.add_argument('--faultFile', default='data/20240311_f/tectonics_Compression.shp', type=str, help='batch size')
parser.add_argument('--valueFile', default='/H5/all_eqA_296_predict_feature.h5', type=str, help='batch size')
parser.add_argument('--value', default='feature', type=str, help='batch size')
parser.add_argument('--isalign', default=False, help='batch size', action='store_true')
parser.add_argument('--isOther', default=False, help='batch size', action='store_true')
parser.add_argument('--swMode', default='single', type=str, help='batch size')
parser.add_argument('--lossFunc', default='Dice', type=str, help='batch size')

args = parser.parse_args()

def get_config(args):
    config = {
        'dataMode': args.dataMode,
        'dataset': args.dataset,
        'gpuIndex': args.gpuIndex,
        'batchsize': args.batchsize,
        'lr0': args.lr0,
        'stepPerMiniBatch': args.stepPerMiniBatch,
        'resDir': args.resDir,
        'mn': args.mn,
        'job': args.job,
        'isAtten': args.isAtten,
        'alpha': args.alpha,
        'dw': args.dw,
        'd0': args.d0,
        'modelType': args.modelType,
        'dropout': args.dropout}
    return config
def write_config(args):
    config = get_config(args)
    keys = ['dataMode', 'dataset', 'gpuIndex', 'batchsize', 'lr0','stepPerMiniBatch', 'resDir', 'mn', 'job', 'isAtten', 'alpha', 'dw', 'd0','dropout', 'modelType']
    with open(config['resDir'] + 'config.txt', 'w') as f:
        for key in keys:
            f.write(key + ': ' + str(config[key]) + '\n')
        
        
