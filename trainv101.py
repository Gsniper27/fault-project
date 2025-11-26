import os
from matplotlib import pyplot as plt
from tool import Generator, data_utils,select_device,findPotential,fault2dis,showPotential,checkPotential,collate_function,RandomScaleRotateBatch,Generator_new,resize_array_with_zoom3d,resize_array_with_zoom,downsample_h5,count_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
#import tensorflow as tf
import platform
import random
#random.seed(0)
import sys
from h5py import File
import numpy as np
import os 
from scipy.ndimage import zoom
from scipy.signal import resample
from scipy import fft
from config import args,write_config
from matplotlib import cm

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
trainFile = 'cascadia_data_to_mark.h5'
dataFile = '/H5/all_eqA_296.h5'
if args.isOther:
    trainFile = 'cascadia_data_to_mark.h5'
#trainFile = '/H5/all_eqA_296_tomark_other.h5'

if args.job =='outputAllKey':
    from glob import glob
    count = 0
    with open('key_all.lst','w') as f:
        for file in glob('outputH5/*.jpg'):
            file = os.path.basename(file)
            f.write(file[:-4]+'\n')
            count +=1
    print(count)
    exit()

if args.job =='outputAllKeyOther':
    from glob import glob
    count = 0
    with open('key_all_new.lst','w') as f:
        for file in glob('outputH5_new/*.jpg'):
            file = os.path.basename(file)
            f.write(file[:-4]+'\n')
            count +=1
    print(count)
    exit()
if args.job =='outputAllKeyOtherSelect':
    from glob import glob
    keys_all = [os.path.basename(file)[:-4] for file in glob('outputH5_new/*.jpg')]
    keys_all_old = [os.path.basename(file)[:-4] for file in glob('outputH5/*.jpg')]
    keys_select_old=  []
    with open('key.lst','r') as f:
        for line in f:
            keys_select_old.append(line.strip())
    with open('key_select_new.lst','w') as f:
        for key in keys_all:
            if key in keys_all_old:
                if key not in keys_select_old:
                    continue
            f.write(key+'\n')
                
    
        

keylist_train = 'cascadia_keys.lst'
keylist_train_added = args.datasetAdd
keyL_h5_added = []
with open(keylist_train,'r') as f:
    keyL_h5 = f.read().split('\n')[:-1]

if keylist_train_added != '':
    with open(keylist_train_added,'r') as f:
        tmp = f.read().split('\n')[:-1]
    for key in tmp:
        if key not in keyL_h5:
            keyL_h5_added.append(key)
print('keyL_h5_added:',len(keyL_h5_added))

random.shuffle(keyL_h5)




dataEnd = np
BatchSize=args.batchsize
inter=1

mn = args.mn.split(',')
n = int(mn[0])
m = int(mn[-1])

strideL = np.zeros([100,2],'int')+1



dataMode = args.dataMode
from numba import njit
from tool import point2length


if __name__ == '__main__':
    
    #from UnetModelTensor import UModel as UnetModel,IoU,backend,keras
    from torchUnet import UNet,norm2,denorm2,huber_loss,torch,WarmupDecayScheduler,IoU,conv_sam,Dice
    
    if args.job == 'extract_value':
        from shapefile import Reader
        
        faultFile = args.faultFile
        
        
        faultFile_value = faultFile.replace('.shp','_value.h5')
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:]
            la = f['la'][:]
            lo = f['lo'][:]
        plt.close()
        plt.figure(figsize=(10,10))
        plt.imshow(data[-1],cmap='bwr')
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.savefig(faultFile_value.replace('.h5',f'_{k0}.jpg'),dpi=300)
        #valueL = []
        with File(faultFile_value,'a') as res:
            with Reader(faultFile) as f:
                for i ,shape in enumerate(f.shapes()):
                    #shape =f.shapes()[1000]
                    print(i,'in',len(f),i/len(f))
                    fault_name  = str(i)
                    points = shape.points
                    points = np.array(points)
                    if fault_name not in res:
                        fault_group = res.create_group(fault_name)
                        fault_group.create_dataset('points',data=points)
                    else:
                        fault_group = res[fault_name]
                    if k0 in fault_group:
                        del fault_group[k0]
                    
                    #import numpy as np

                    # 假设 la 和 lo 是升序排列的
                    I = np.searchsorted(la, points[:,1])
                    J = np.searchsorted(lo, points[:,0])
                    
                    

                    # 限制 I, J 在合法范围内
                    I = np.clip(I, 0, len(la) - 3)
                    J = np.clip(J, 0, len(lo) - 3)

                    #value = data[:, I, J]
                    # 先获取所有可能的索引组合
                    #value = np.take(np.take(data, I, axis=1), J, axis=2)
                    # 生成 I, J 的唯一组合
                    #unique_pairs, unique_idx = np.unique(np.stack([I, J], axis=1), axis=0, return_inverse=True)
                    #unique_I, unique_J = unique_pairs[:, 0], unique_pairs[:, 1]

                    # 只查询唯一的 I, J 组合，减少重复访问
                    #unique_values = data[:, unique_I, unique_J]

                    # 还原回原来的索引顺序
                    #value = unique_values[:, unique_idx]
                    #I  = np.abs(la[np.newaxis]-points[:,1][:,np.newaxis]).argmin(axis=1)
                    #J  = np.abs(lo[np.newaxis]-points[:,0][:,np.newaxis]).argmin(axis=1)
                    #print(I,J,points[:,0],points[:,1],la[::100],data.shape)
                    #exit()
                    value = data[:,I,J]/4+data[:,I+1,J+1]/4+data[:,I+1,J]/4+data[:,I,J+1]/4
                    #value1 = data[:,J,I]
                    #print(value.max(),value1.max())#,I,J,points[:,0])
                    print(la.shape,lo.shape,data.shape)
                    fault_group.create_dataset(k0,data=value.astype('float16'))
                    
                    #exit()
                    #valueL.append(value.mean(axis=1))
            #valueL = np.stack(valueL,axis=0)
            #np.save(faultFile_value,valueL)
        exit()
    if args.job == 'show_value':
        from shapefile import Reader
        file = args.valueFile
        faultFile = args.faultFile
        k0 = args.value
        
        faultFile_value = faultFile.replace('.shp','_value.h5')
        
        plotDir = faultFile.replace('.shp','/')
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        
        with File(faultFile_value,'r') as f:
            for k in list(f.keys())[::100]:
                group = f[k]
                print(group)
                value = group[k0][:]/255
                points = group['points'][:]
                plt.close()
                plt.figure(figsize=(10,10))
                plt.plot(points[:,0],points[:,1],'k')
                plt.scatter(points[:,0],points[:,1],c=value,cmap='bwr')
                plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.savefig(f'{plotDir}{k}.jpg',dpi=300)
                
        exit()
    if args.job == 'show_tsne':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        compressionFile  = 'data/20240311_f/tectonics_Compression.h5'
        extensionFile  = 'data/20240311_f/tectonics_Extention.h5'
        featureL_compression = []
        minN = 30
        plotDir = args.resDir+'tsne/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        with File(compressionFile,'r') as f:
            for k in list(f.keys())[::1]:
                #print(f[k].keys())
                if f[k]['feature'].shape[1]<minN:
                    continue
                featureL_compression.append(f[k]['feature'][:].mean(axis=1))
        featureL_compression = np.stack(featureL_compression,axis=0)
        featureL_extension = []
        with File(extensionFile,'r') as f:
            for k in list(f.keys())[::1]:
                if f[k]['feature'].shape[1]<minN:
                    continue
                featureL_extension.append(f[k]['feature'][:].mean(axis=1))
        featureL_extension = np.stack(featureL_extension,axis=0)
        featureL = np.concatenate([featureL_compression,featureL_extension],axis=0)
        labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]])])
        featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
        featureL_norm = featureL_norm /((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
        tsne = TSNE(n_components=2,random_state=0)
        #tsne = PCA(n_components=2)
        x = tsne.fit_transform(featureL_norm)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.05,label='e',mec='None')
        
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne.jpg',dpi=600)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.0,label='e',mec='None')
        #plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_c.jpg',dpi=600)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.0,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.05,label='e',mec='None')
        #plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_e.jpg',dpi=600)
        exit()
    if args.job == 'show_feature':
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:,::2,::2]
            la = f['la'][::2]
            lo = f['lo'][::2]
        plotDir = args.resDir+'features/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        for i in range(len(data)):
            print(i)
            plt.close()
            plt.figure(figsize=(10,10))
            #plt.imshow(data[i],cmap='jet')
            plt.pcolormesh(lo,la,data[i],cmap='jet',rasterized=True)
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.savefig(f'{plotDir}/{i}.jpg',dpi=300)
    if args.job == 'cal_feature_corr':
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:,::4,::4]
            la = f['la'][::4]
            lo = f['lo'][::4]
        corr = np.zeros([data.shape[0],data.shape[0]])
        
        data  =torch.from_numpy(data).float()
        data_norm = (data-data.mean(axis=(1,2),keepdims=True))/(data.std(axis=(1,2),keepdims=True)+1e-4)
        data_norm = data_norm.to('cuda')
        
        for i in range(data.shape[0]):
            for j in range(i,data.shape[0]):
                data0 = data_norm[i]#.to('cuda')
                data1 = data_norm[j]#.to('cuda')
                Corr = (data0*data1).mean().cpu().numpy()
                corr[i,j] = Corr
                corr[j,i] = Corr
                print(i,j,corr[i,j])
        plt.close()
        plt.figure(figsize=(10,10))
        plt.imshow(corr,cmap='bwr',vmin=-1,vmax=1)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.savefig(f'{args.resDir}corr.jpg',dpi=300)
        np.savetxt(f'{args.resDir}corr.txt',corr)
    if args.job == 'an_feature_corr':
        corr = np.loadtxt(f'{args.resDir}corr.txt')
        
        PCA = np.linalg.eig(corr)
        threshod = PCA[0].max()*0.005
        plt.close()
        plt.figure(figsize=(10,10))
        plt.plot(PCA[0])
        plt.plot([0,256],[threshod,threshod],
                 'k')
        plt.savefig(f'{args.resDir}PCA.jpg',dpi=300)
        print((PCA[0]>threshod).sum())
        exit()
    if args.job == 'show_feature_corr':
        corr = np.loadtxt(f'{args.resDir}corr.txt')
        PCA = np.linalg.eig(corr)
        k0 = args.value
        file = args.valueFile
        plotDir = args.resDir+'features/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        with File(file,'r') as f:
            data = f[k0][:,::4,::4]
            la = f['la'][::4]
            lo = f['lo'][::4]
        threshod = PCA[0].max()*0.005
        data  =torch.from_numpy(data).float()
        data_norm = (data-data.mean(axis=(1,2),keepdims=True))/(data.std(axis=(1,2),keepdims=True)+1e-4)
        data_norm = data_norm.to('cuda')
        for i in range((PCA[0]>threshod).sum()):
            v = PCA[1][i]
            v = torch.from_numpy(v).float().to('cuda')
            feature = torch.einsum('kij,k->ij',data_norm,v).cpu().numpy()
            feature = feature/feature.std()
            plt.close()
            plt.figure(figsize=(10,5))
            #plt.imshow(data[i],cmap='jet')
            plt.pcolormesh(lo,la,feature,cmap='RdYlBu',rasterized=True,vmin=-3,vmax=3)
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.savefig(f'{plotDir}/feature_{i}.jpg',dpi=300)
    if args.job in ['show_kmean','show_kmean_o','show_kmean_river']:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        from scipy import io as sio
        if args.job ==  'show_kmean':
            compressionFile  = 'data/0zero9/tectonics_Compression_value.h5'
            extensionFile  = 'data/0zero9/tectonics_Extention_value.h5'
            #little_riverFile = 'data/river/Hynek-valley_value.h5'
            plotDir = args.resDir+'tsne/'
        elif args.job == 'show_kmean_river':
            compressionFile  = 'data/20240311_f/tectonics_Compression_value.h5'
            extensionFile  = 'data/20240311_f/tectonics_Extention_value.h5'
            little_riverFile = 'data/river/Hynek-valley_value.h5'
            large_riverFile = 'data/fluvial_tanaka/fluvial_tanaka2014_value.h5'
            plotDir = args.resDir+'tsne_river/'
        else:
            compressionFile  = 'data/tectonics_Compression_value.h5'
            extensionFile  = 'data/tectonics_Extention_value.h5'
            plotDir = args.resDir+'tsne_o/'
        featureL_compression = []
        lolaL_compression = []
        pointsL_compression = []
        distanceL_compression = []
        minN = 10
        minDis = 0.2
        
        
        
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
            
            
            
        from shapefile import Reader
        vnsla = []
        vnslo = []
        with Reader('data/VNs/valley.shp') as f:
            for i,shape in enumerate(f.shapes()):
                points = np.array(shape.points)
                vnsla.append(points[:,1])
                vnslo.append(points[:,0])
                vnsla.append(np.ones(1)*np.nan)
                vnslo.append(np.ones(1)*np.nan)
        vnsla = np.concatenate(vnsla)
        vnslo = np.concatenate(vnslo)
        
        plt.close()
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.savefig(f'{plotDir}/vns.jpg',dpi=600)
        
        with File(compressionFile,'r') as f:
            for k in list(f.keys())[::1]:
                #print(f[k].keys())
                if f[k]['feature'].shape[1]<minN:
                    continue
                distance = point2length(f[k]['points'][:])
                if distance.min()<minDis:
                    continue
                featureL_compression.append(f[k]['feature'][:].mean(axis=1))
                lolaL_compression.append(f[k]['points'][:].mean(axis=0))
                pointsL_compression.append(f[k]['points'][:])
                distanceL_compression.append(distance)
        featureL_compression = np.stack(featureL_compression,axis=0)
        lolaL_compression = np.stack(lolaL_compression,axis=0)
        distanceL_compression = np.stack(distanceL_compression,axis=0)
        
        featureL_extension = []
        lolaL_extension = []
        pointsL_extension = []
        distanceL_extension = []
        
        with File(extensionFile,'r') as f:
            for k in list(f.keys())[::1]:
                if f[k]['feature'].shape[1]<minN:
                    continue
                distance = point2length(f[k]['points'][:])
                if distance.min()<minDis:
                    continue
                featureL_extension.append(f[k]['feature'][:].mean(axis=1))
                lolaL_extension.append(f[k]['points'][:].mean(axis=0))
                pointsL_extension.append(f[k]['points'][:])
                distanceL_extension.append(distance)
        featureL_extension = np.stack(featureL_extension,axis=0)
        lolaL_extension = np.stack(lolaL_extension,axis=0)
        if args.job == 'show_kmean_river':
            featureL_little_river = []
            lolaL_little_river = []
            pointsL_little_river = []
            distanceL_little_river = []
            with File(little_riverFile,'r') as f:
                for k in list(f.keys())[::1]:
                    #print(f[k].keys())
                    if f[k]['feature'].shape[1]<minN:
                        continue
                    distance = point2length(f[k]['points'][:])
                    if distance.min()<minDis:
                        continue
                    featureL_little_river.append(f[k]['feature'][:].mean(axis=1))
                    lolaL_little_river.append(f[k]['points'][:].mean(axis=0))
                    pointsL_little_river.append(f[k]['points'][:])
                    distanceL_little_river.append(distance)
            featureL_little_river = np.stack(featureL_little_river,axis=0)
            lolaL_little_river = np.stack(lolaL_little_river,axis=0)
            distanceL_little_river = np.stack(distanceL_little_river,axis=0)
            
            
            featureL_large_river = []
            lolaL_large_river = []
            pointsL_large_river = []
            distanceL_large_river = []
            with File(large_riverFile,'r') as f:
                for k in list(f.keys())[::1]:
                    #print(f[k].keys())
                    if f[k]['feature'].shape[1]<minN:
                        continue
                    distance = point2length(f[k]['points'][:])
                    if distance.min()<minDis:
                        continue
                    featureL_large_river.append(f[k]['feature'][:].mean(axis=1))
                    lolaL_large_river.append(f[k]['points'][:].mean(axis=0))
                    pointsL_large_river.append(f[k]['points'][:])
                    distanceL_large_river.append(distance)  
                    
            featureL_large_river = np.stack(featureL_large_river,axis=0)
            lolaL_large_river = np.stack(lolaL_large_river,axis=0)
            distanceL_large_river = np.stack(distanceL_large_river,axis=0)
            
            
            
        
        if args.job == 'show_kmean_river':
            featureL = np.concatenate([featureL_compression,featureL_extension,featureL_little_river,featureL_large_river],axis=0)
            labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]]),np.ones([featureL_little_river.shape[0]])*2,np.ones([featureL_large_river.shape[0]])*3])
            lolaL = np.concatenate([lolaL_compression,lolaL_extension,lolaL_little_river,lolaL_large_river],axis=0)
            distanceL_extension= np.stack(distanceL_extension,axis=0)
            distanceL = np.concatenate([distanceL_compression,distanceL_extension,distanceL_little_river,distanceL_large_river],axis=0)
            pointsL = pointsL_compression+pointsL_extension+pointsL_little_river+pointsL_large_river
            featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
            featureL = featureL_norm/((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
            
        else:
            featureL = np.concatenate([featureL_compression,featureL_extension],axis=0)
            labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]])])
            lolaL = np.concatenate([lolaL_compression,lolaL_extension],axis=0)
            distanceL_extension= np.stack(distanceL_extension,axis=0)
            distanceL = np.concatenate([distanceL_compression,distanceL_extension],axis=0)
            pointsL = pointsL_compression+pointsL_extension
            featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
            featureL = featureL_norm/((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
        
        from sklearn.cluster import KMeans
        
        cN = 3
        colorL  = ['r','g','b','c','m','y','k','brown']
        kmeans = KMeans(n_clusters=cN, random_state=0).fit(featureL)
        import shapefile
        shpDir = args.resDir+'shp/'
        if not os.path.exists(shpDir):
            os.makedirs(shpDir)
        
        sf_compression = shapefile.Writer(shpDir+'compression.shp', shapeType=shapefile.POLYLINE)
        sf_extension = shapefile.Writer(shpDir+'extension.shp', shapeType=shapefile.POLYLINE)
        sf_compression.field('class', "N", size=5) 
        sf_extension.field('class', "N", size=5)
        lines_compression = []
        lines_extension = []
        for i in range(len(featureL)):
            line = [[[point[0],point[1]]for point in pointsL[i]]]
            Class = kmeans.labels_[i]
            
            if i<len(pointsL_compression):
                sf = sf_compression
            else:
                sf = sf_extension
            sf.line(line)
            sf.record(Class)
        
        sf_compression.close()
        sf_extension.close()
        #tsne = PCA(n_components=2)
        tsne = TSNE(n_components=3,random_state=0)
        x = tsne.fit_transform(featureL)
        dataD = {
            'x':x,
            'label':kmeans.labels_,
            'lola':lolaL,
            'points':pointsL,
            'feature':featureL,
            'ce_label':labelL,
            'distance':distanceL
        } 
        sio.savemat(f'{args.resDir}tsne.mat',dataD)
        
        
        
        tsne = TSNE(n_components=2,random_state=0)
        x = tsne.fit_transform(featureL)
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.k',ms=1,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):len(featureL_compression)+len(featureL_extension),0],x[len(featureL_compression):len(featureL_compression)+len(featureL_extension),1],'.b',ms=1,alpha=0.5,label='e',mec='None')
        if args.job == 'show_kmean_river':
            plt.plot(x[len(featureL_compression)+len(featureL_extension):len(featureL_compression)+len(featureL_extension)+len(featureL_little_river),0],x[len(featureL_compression)+len(featureL_extension):len(featureL_compression)+len(featureL_extension)+len(featureL_little_river),1],'.r',ms=1,alpha=0.5,label='little river',mec='None')
            plt.plot(x[len(featureL_compression)+len(featureL_extension)+len(featureL_little_river):,0],x[len(featureL_compression)+len(featureL_extension)+len(featureL_little_river):,1],'.g',ms=4,alpha=0.5,label='large river',mec='None')
        
        plt.gca().set_aspect('equal')
        #plt.legend()
        plt.savefig(f'{plotDir}/tsne.jpg',dpi=600)
        plt.close()
        
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_==i,0],x[kmeans.labels_==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean.jpg',dpi=600)
        
        x = tsne.fit_transform(featureL[:len(featureL_compression)])
        plt.close()
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_[:len(featureL_compression)]==i,0],x[kmeans.labels_[:len(featureL_compression)]==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean_c.jpg',dpi=600)
        
        
        x = tsne.fit_transform(featureL[len(featureL_compression):])
        plt.close()
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_[len(featureL_compression):]==i,0],x[kmeans.labels_[len(featureL_compression):]==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean_e.jpg',dpi=600)
                
        plt.close()
        plt.figure(figsize=(15,10))
        #for i in range(cN):   
        #    #plt.plot(lolaL[kmeans.labels_==i,0],lolaL[kmeans.labels==i,1],'.',ms=1,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        for i in range(len(pointsL)):
            plt.plot(pointsL[i][:,0],pointsL[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i]])
        plt.gca().set_aspect('equal')
        #plt.legend()
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean.jpg',dpi=600)
        plt.xlim([40,65])
        plt.ylim([-35,-15])
        plt.savefig(f'{plotDir}/kmean_vns.jpg',dpi=600)
        
        plt.close()
        plt.figure(figsize=(15,10))
        for i in range(len(pointsL_compression)):
            plt.plot(pointsL_compression[i][:,0],pointsL_compression[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i]])
        plt.gca().set_aspect('equal')
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean_c.jpg',dpi=600)
        
        plt.close()
        plt.figure(figsize=(15,10))
        for i in range(len(pointsL_extension)):
            plt.plot(pointsL_extension[i][:,0],pointsL_extension[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i+len(pointsL_compression)]])   
        plt.gca().set_aspect('equal')
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean_e.jpg',dpi=600)
        exit()
        
        
        
    if args.job == 'downSamplingO':
        h5File_in = '/H5/all_eqA_296.h5'
        h5File = '/H5/all_eqA_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16 
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File_downsample = '/NAS/jyr/mount/bak/bakdata/data/all_eqA_296_predict_ds.h5'
        #pltDir  = plotDir+'show_all/'
        #if not os.path.exists(pltDir):
        #    os.makedirs(pltDir)
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        if True:           
            downsample_h5(h5File,h5File_downsample,k0,factor,'max')
            downsample_h5(h5File,h5File_downsample,k1,factor,'max')
            downsample_h5(h5File,h5File_downsample,'image',factor)
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            f['la'] = laL[::factor]/2+laL[::-factor][::-1]/2
            f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2
    if args.job == 'downSampling':
        h5File_in = '/H5/all_eqA_296.h5'
        h5File = '/H5/all_eqA_296_predict.h5'
        
        #h5File_in = '/data/jiangyr/data/all_296.h5'
        #h5File = '/data/jiangyr/data/all_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File_downsample = f'/H5/all_296_predict_ds_{factor}.h5'
        pltDir  = args.resDir+'show_all/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        #if False:
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]    
        if False :    
            downsample_h5(h5File_in,h5File_downsample,k0,factor,dataset_name_out=k0+'_o',dtype='uint8',d0=args.d0,method='max')
            downsample_h5(h5File_in,h5File_downsample,k1,factor,dataset_name_out=k1+'_o',dtype='uint8',d0=args.d0,method='max')
            #exit() 
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            NLa = laL.shape[0]//factor
            NLo = loL.shape[0]//factor
            f['la'] = (laL[::factor]/2+laL[::-factor][::-1]/2)[:NLa]#resample(laL,laL.shape[0]//factor)
            f['lo'] = (loL[::factor]/2+loL[::-factor][::-1]/2 )[:NLo]   #resample(loL,loL.shape[0]//factor)
            #laL[::factor]/2+laL[::-factor][::-1]/2
            #f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2      
            print(f['la'][:])
            print(f['lo'][:])
            #exit()
        downsample_h5(h5File_in,h5File_downsample,k0,factor,dataset_name_out=k0+'_o',dtype='uint8',d0=args.d0,method='max')
        downsample_h5(h5File_in,h5File_downsample,k1,factor,dataset_name_out=k1+'_o',dtype='uint8',d0=args.d0,method='max')
        downsample_h5(h5File_in,h5File_downsample,'TW',factor,dataset_name_out='image_o',)
        downsample_h5(h5File,h5File_downsample,k0,factor,'max')
        downsample_h5(h5File,h5File_downsample,k1,factor,'max')
        downsample_h5(h5File,h5File_downsample,'image',factor)
        
        #if False:
    if args.job == 'show_all_ds_compare':
        import matplotlib.colors as mcolors
        factor = 16
        h5File = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File = f'/H5/all_296_predict_ds_{factor}.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 20
        size = 2048
        pltDir  = args.resDir+'show_all_ds/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        with File(h5File,'r') as f:
            la = f['la'][:]
            lo = f['lo'][:]
            print(la)
            print(lo)
            #exit()
            data0 = f[k0][:].astype('float32')
            data1 = f[k1][:].astype('float32')
            data0_o = f[k0+'_o'][:].astype('float32')
            data1_o = f[k1+'_o'][:].astype('float32')
            image = f['image_o'][:].astype('float32')/255
            image_out = f['image'][:].astype('float32')/255
            image_gray = image*0+image.mean(axis=0,keepdims=True)
            image_gray= 1-(1-image_gray)**2
            #image[:] = image.mean(axis=0,keepdims=True)#*0.5
            #print(image.shape,image.min(),image.max())  
            #image = 1-(1-image)**2  
            #exit()
            threshod = 255*args.threshold
            data0[data0<=threshod]=np.nan
            data1[data1<=threshod]=np.nan
            data0_o[data0_o<=threshod]=np.nan
            data1_o[data1_o<=threshod]=np.nan
            
            data0[data0>threshod]=255
            data1[data1>threshod]=255
            data0_o[data0_o>threshod]=255
            data1_o[data1_o>threshod]=255
            
            data0_o_ = data0_o#.copy()
            #data0_o_[::2]=np.nan
            
            data1_o_ = data1_o#.copy()
            #data1_o_[::2]=np.nan
            plt.figure(figsize=(20,10))
            
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data0_o_[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=0.2)
            
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}ext_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o_[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=0.2)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}comp_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image_out[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_out[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image_out[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_out[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            polarIndex = np.abs(la+75).argmin()
            print(polarIndex)
            laL_polar = la[:polarIndex]
            laL_polar_rad = np.deg2rad(90+laL_polar)
            loL_polar = lo[:]
            loL_polar_rad = np.deg2rad(loL_polar)
            
            
            
            x = np.cos(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            y = np.sin(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            #ax.pcolormesh(x,y,image[:,:polarIndex].transpose([1,2,0]),rasterized=True,alpha=1,)
            z = image_out[:,-polarIndex:][:,::-1][:,:-1,:-1].transpose([1,2,0])
            colors_image = z.reshape([-1,3]) 
            colors_TW = f['image_o'][:,polarIndex:][:,::-1][:,:-1,:-1].transpose([1,2,0]).reshape([-1,3]).astype('float32')/255
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_image)
            ax.pcolorfast(x,y,data0[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolorfast(x,y,data1[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_north_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_TW)
            ax.pcolormesh(x,y,data0[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_north_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            x = np.cos(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            y = np.sin(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            plt.figure(figsize=(10,10),dpi=600,)
            z = image_out[:,:polarIndex][:,:-1,:-1].transpose([1,2,0])
            colors_image = z.reshape([-1,3])  
            colors_TW = f['image_o'][:,:polarIndex][:,:-1,:-1].transpose([1,2,0]).reshape([-1,3]).astype('float32')/255
            
            fig, ax = plt.subplots()
            #ax.pcolormesh(x,y,image[:,:polarIndex].transpose([1,2,0]),rasterized=True,alpha=1,)
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_image )
            ax.pcolormesh(x,y,data0[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            #exit()
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_TW)
            ax.pcolormesh(x,y,data0[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            exit()
            
    if args.job == 'downSamplingIn':
        h5File = '/H5/all_eqA_296.h5'
        #h5File = '/H5/all_eqA_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16 
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_ds_{factor}.h5'
        #h5File_downsample = '/NAS/jyr/mount/bak/bakdata/data/all_eqA_296_predict_ds.h5'
        #pltDir  = plotDir+'show_all/'
        #if not os.path.exists(pltDir):
        #    os.makedirs(pltDir)
        with File(h5File,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        if True:           
            downsample_h5(h5File,h5File_downsample,k0,factor,'max')
            downsample_h5(h5File,h5File_downsample,k1,factor,'max')
            downsample_h5(h5File,h5File_downsample,'image',factor)
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            f['la'] = laL[::factor]/2+laL[::-factor][::-1]/2
            f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2
            
    if args.job=='toDis' :
        fault2dis(dataFile,'tectonics_Extention.shp',maxDis=0.4,head='Extention')
        fault2dis(dataFile,'tectonics_Compression.shp',maxDis=0.4,head='Compression')
        exit()
    
    #   ['graben', 'flow-like features', 'channel', 'river channel', 'wrinkle-like features', 'aeolian Yardang landform', 'glacial landform', 'chaotic terrain']
    if args.job=='toDis_new' :
        keys = [ 'flow-like features', 'channel', 'river channel', 'wrinkle-like features', 'aeolian Yardang landform', 'glacial landform']
        fault2dis(dataFile,'data/AI_extention_YWPick/tectonics_Extention_0.95_faults.shp',maxDis=0.4,head='others',keys=keys)
        #fault2dis(dataFile,'tectonics_Compression.shp',maxDis=0.4,head='Compression')
        #exit()
    if args.job == 'plotDis' :
        with File(dataFile,'r') as h5:
            image = h5['tectonics_Extention_0.95_faults'][::10,::10]
            image = np.where(image<0.35,image,np.nan)
            plt.imshow(image,cmap='bwr')
            plt.colorbar()
            plt.savefig('tectonics_Extention_0.95_faults.png')
            plt.show()
            exit()
            
    if args.job=='findP' :
        findPotential(dataFile,d=n)
        exit()
    if args.job=='showP' :
        showPotential(dataFile)
    if args.job=='checkP' :
        checkPotential(dataFile)
    

    device = 'cpu' if (args.gpuIndex < 0 or not torch.cuda.is_available()) else f'cuda:{args.gpuIndex}'
    
    
    if args.modelType=='mars_sam':
        net = conv_sam(32,dropout=args.dropout,isvit=False,isalign=args.isalign)
    elif args.modelType=='mars_sam_vit':
        net = conv_sam(32,dropout=args.dropout,isvit=True,isalign=args.isalign)
    elif args.modelType=='mars_sam_re':
        net = conv_sam(32,reRandom=True,dropout=args.dropout,)
    else:
        net =UNet(32,[1,2,2,4,8],[3,4],2,args.dropout,8,7,m,n,3)
        
    predictFile = f'/data/jiangyr/data/all_eqA_296_predict_{args.modelType}.h5'
    predictFile = f'/H5/all_eqA_296_predict.h5'
    net.to(device)
    
    step_per_batches = args.stepPerMiniBatch
    if args.modelType in ['mars_sam','mars_sam_vit']:
        opt = torch.optim.AdamW(
        [
            {"params": net.not_sam.parameters(),'lr':args.lr0},
            {"params": net.in_sam.parameters(),'lr':args.lr0/8}, 
        ],
            lr=args.lr0
            )
        optSGD = torch.optim.SGD(
        [
            {"params": net.not_sam.parameters(),'lr':args.lr0*10},
            {"params": net.in_sam.parameters(),'lr':args.lr0}, 
        ],
            lr=args.lr0
            )
    else:
        opt = torch.optim.AdamW(
        [
            {"params": net.parameters(),'lr':args.lr0},
            
        ],
            lr=args.lr0
            )
    #scheduler = WarmupDecayScheduler(opt, warmup_steps//step_per_batches, total_steps//step_per_batches, initial_lr, final_lr, decay_type='exponential')
    if args.job=='showModel':
        net.summary()
        exit()
    
    dataSetTrain = Generator_new(trainFile,keyL_h5[:len(keyL_h5)//10*9]+keyL_h5_added*32,m=m,n=n,training=True)
    dataSetTest = Generator_new(trainFile,keyL_h5[len(keyL_h5)//10*9:],m=m,n=n,training=True)
    
    train_loader_train = data_utils.DataLoader(dataSetTrain,
                                    batch_size=BatchSize,
                                    shuffle=True, drop_last=True, num_workers=0,collate_fn=collate_function,)
   

    d0 = args.d0
    dw =args.dw
    #D = 0.
    alpha = args.alpha
    resDir = args.resDir
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    plotDir = resDir+'plot/'
    modelFile = f'{resDir}/model.pth'
    h5FileNew =resDir+'new.h5'

    if args.job=='showSTD':
        for j in range(10):
            image,te,tc,scale=dataSetTest[0]
            #print(image.std(axis=(0,2,3))) 
            #print(image.mean(axis=(0,2,3)))
            #print('____')
            plt.close()
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.pcolor(image[0,:,:])
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.pcolor(image[1,:,:])
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{plotDir}{j}.jpg',dpi=300)
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    if args.job=='train':
        calEnd = torch
    if args.lossFunc =='IoU':
        lossFunc = IoU
    elif args.lossFunc =='Dice':
        lossFunc = Dice
    write_config(args)

    # --- SETUP FOR EARLY STOPPING AND SCHEDULER ---
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
    best_val_loss = float('inf')
    patience_counter = 0
    patience_epochs = 5  # Stop training if the loss doesn't improve for 5 epochs
    # --- END OF SETUP ---

    count_parameters_all, count_parameters_train = count_parameters(net)
    with open(f'{resDir}parameters.txt','a') as f:
        f.write(f'{count_parameters_all} {count_parameters_train}\n')

    # --- START OF NEW EPOCH LOOP (UP TO 100) ---
    for i in range(100):
        epoch_loss = 0.0
        batches = 0

        net.train()
        if 'sam' in args.modelType:
            net.in_sam.eval()

        print(f"--- Starting Epoch {i} ---")
        # --- Your original training batch loop ---
        for j, (image,target) in enumerate(train_loader_train):
            batches += 1

            # --- Learning rate warmup for epoch 0 ---
            if i==0 and j < len(train_loader_train): # Check j to avoid division by zero if loader is empty
                lr_update = args.lr0 * (1 - np.exp(-j / len(train_loader_train) * 3))
                opt.param_groups[0]['lr'] = lr_update
                if 'sam' in args.modelType and len(opt.param_groups) > 1:
                    opt.param_groups[1]['lr'] = lr_update / 8

            if not isinstance(image,torch.Tensor):
                image =torch.from_numpy(image).to(device)
                target = torch.from_numpy(target).to(device)

            target,others =target[:,:2],target[:,2:]

            if args.swMode=='single':
                tMin = target+0
            else:
                tMin = target.min(dim=1,keepdim=True).values
            sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha

            if args.isOther:
                sw_other = (calEnd.exp(-others**2/(dw)**2)*(1-alpha)+alpha)
                sw_other = sw_other.expand_as(sw)
                sw = torch.where(sw>sw_other,sw,sw_other)
            else:
                sw_other = torch.zeros_like(sw) # Placeholder

            target = calEnd.where(target<d0,1.,0.)
            valid_01 = image[:,2:4].amin(dim=1,keepdim=True)
            sw = calEnd.where(valid_01<1e-9,1,sw)

            sw = sw/sw.mean()

            inputs = image
            predict = net(inputs)

            loss = lossFunc(target,predict,sw=sw)
            epoch_loss += loss.item()

            print(f"Epoch {i} [Batch {j+1}/{len(train_loader_train)}]: Loss = {loss.item():.6f}", end='\r')

            loss = loss/step_per_batches
            loss.backward()

            if (j+1) % step_per_batches == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                opt.step()
                opt.zero_grad()

        avg_train_loss = epoch_loss / max(batches, 1)
        print() # New line after the training epoch finishes

        # --- START OF NEW VALIDATION & EARLY STOPPING LOGIC ---
        print("Running validation...")
        with torch.no_grad():
            net.eval()
            lossL =[]

            # --- Create a separate validation data loader ---
            # It's better to not re-use the training loader for validation
            train_loader_test = data_utils.DataLoader(dataSetTest,
                                                batch_size=BatchSize,
                                                shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_function)

            for j, (image,target) in enumerate(train_loader_test): 
                if not isinstance(image,torch.Tensor):
                    image =torch.from_numpy(image).to(device)
                    target = torch.from_numpy(target).to(device)

                target,others = target[:,:2],target[:,2:]
                if args.swMode=='single':
                    tMin = target+0
                else:
                    tMin = target.min(dim=1,keepdim=True).values
                sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha

                if args.isOther:
                    sw_other = (calEnd.exp(-others**2/(dw)**2)*(1-alpha)+alpha)
                    sw_other = sw_other.expand_as(sw)
                    sw = torch.where(sw>sw_other,sw,sw_other)
                else:
                    sw_other = torch.zeros_like(sw)

                target_binary = calEnd.where(target<d0,1.,0.) # Use a different name
                valid_01 = image[:,2:4].min(dim=1,keepdim=True).values
                sw = calEnd.where(valid_01<1e-9,1,sw)
                sw = sw/sw.mean()

                outputs = net(image)
                loss = lossFunc(target_binary,outputs,sw=sw).detach().cpu().item()
                lossL.append(loss)

                # --- NEW PLOTTING LOGIC ---
                # Only save ONE plot, and ONLY if it's a useful, positive sample
                if j == 0: 
                    for k in range(1): # Only check the first image (k=0)
                        if np.sum(target_binary.cpu().numpy()[k]) > 0: # Check if the target has any faults
                            # --- Your original plotting code, now safely indented ---
                            output_image = image[:, -3:, :, :].detach().cpu().numpy()
                            if args.isalign:
                                inputs_shift = net.align(image).detach().cpu().numpy()
                            else:
                                inputs_shift= image.detach().cpu().numpy()
                            inputs = image.detach().cpu().numpy()
                            output = outputs.detach().cpu().numpy()
                            target_plot = target_binary.cpu().detach().numpy() # Use the binary target
                            sw_plot = sw.cpu().detach().numpy()
                            sw_other_plot = sw_other.cpu().detach().numpy()
                            target_plot[target_plot<0.5]=np.nan
                            output[output<0.5]=np.nan

                            plt.close()
                            plt.figure(figsize=(10,15))
                            plt.subplot(3,2,1)
                            plt.imshow(inputs[k,0],cmap='gray')
                            plt.imshow(target_plot[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                            plt.title('Extention target')
                            plt.gca().set_aspect('equal')

                            plt.subplot(3,2,2)
                            plt.imshow(inputs[k,1],cmap='gray')
                            plt.imshow(target_plot[k,1],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                            plt.title('Compression target')
                            plt.gca().set_aspect('equal')

                            plt.subplot(3,2,3)
                            plt.imshow(np.clip(inputs[k,4:].transpose([1,2,0])/2+0.5,0,1))
                            plt.imshow(output[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                            plt.title('Extention predict')
                            plt.gca().set_aspect('equal')
                            plt.subplot(3,2,4)
                            plt.imshow(np.clip(output_image[k].transpose([1,2,0]),0,1))
                            plt.imshow(output[k,1],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                            plt.title('Compression predict')
                            plt.tight_layout()

                            plt.subplot(3,2,5)
                            plt.imshow(sw_plot[k,0],cmap='gray')
                            plt.title('weight')

                            plt.subplot(3,2,6)
                            plt.imshow(sw_other_plot[k,0],cmap='gray')
                            plt.title('weight other')

                            plt.tight_layout()
                            plt.savefig(f'{plotDir}{i}_{k}.jpg',dpi=300)
                            # --- End of original plotting code ---

        avg_val_loss = np.mean(lossL)
        print(f"Epoch {i}: Avg Train Loss = {avg_train_loss:.6f}, Avg Validation Loss = {avg_val_loss:.6f}")

        with open(f'{resDir}loss.txt','a') as f:
            f.write(f'{i} {avg_train_loss} {avg_val_loss}\n')

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), modelFile) # Save the new best model
            print(f"  -> Validation loss improved. Saved new best model to {modelFile}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{patience_epochs}")

        if patience_counter >= patience_epochs:
            print(f"\nEarly stopping triggered after {i} epochs. Best validation loss: {best_val_loss:.6f}")
            break # Exit the main training loop

# --- END OF UPGRADED TRAINING BLOCK ---

# ... (Keep all your other job blocks like 'predict_test', 'predict_all', etc. exactly as they were)

keyL =['tectonics_Extention','tectonics_Compression']
if  args.job=='predict_test' :
    from PIL import Image
    calEnd = torch
    net.load_state_dict(torch.load(modelFile,map_location=torch.device('cpu')))
    dataSetTest = Generator_new(trainFile,keyL_h5[len(keyL_h5)//10*9:],m=m,n=n)

    train_loader_test = data_utils.DataLoader(dataSetTest,
                                        batch_size=BatchSize,
                                        shuffle=False, drop_last=True, num_workers=0,collate_fn=collate_function)
    #print(len(train_loader_test))
    plotDir=resDir+'plot_test/'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    for j, (image,target) in enumerate(train_loader_test):

        if not isinstance(image,torch.Tensor):
            image =torch.from_numpy(image).to(device)
            target = torch.from_numpy(target).to(device)


        if not isinstance(image,torch.Tensor):
                    image =torch.from_numpy(image).to(device)
                    target = torch.from_numpy(target).to(device)


        tMin = target.min(dim=1,keepdim=True).values


        target = calEnd.where(target<d0,1.,0.)


        sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha
        sw = sw/sw.mean()

        outputs = net.Predict(image)
        output_image = net.outputImage(image).detach().cpu().numpy()
        inputs = image.detach().cpu().numpy()
        if args.isalign:
            inputs_shift = net.align(image).detach().cpu().numpy()
        else:
            inputs_shift = inputs
        #theta0,theta1 = net.align.get_theta(image)
        #print(theta0[0].detach().cpu().numpy())
        #print(theta1[0].detach().cpu().numpy())
        #exit()
        output = outputs
        #output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        target[target<0.5]=np.nan
        #target[target>0.5]=0.5
        output[output<0.5]=np.nan
        #output[output>0.5]=0.5
        #inputs = inputs.cpu().detach().numpy()

        for k in range(target.shape[0]):

            if False:
                print(convertImage[k].shape)
                image = Image.fromarray(convertImage[k])
                image.save(f'{plotDir}image_{j}_{k}.jpg')
                plt.close()
                #plt.figure(figsize=(5,16))
                plt.figure(figsize=(6,6))
                plt.subplot(2,2,1)
                plt.imshow(inputs[k,0],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                plt.imshow(target[k,0],cmap='jet',vmin=0,vmax=1,rasterized=True,alpha=0.4)
                plt.title('CTX target')
                #plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.subplot(2,2,2)

                plt.imshow(inputs[k,1],cmap='jet',rasterized=True)
                plt.imshow(target[k,1],cmap='jet',vmin=0,vmax=1,rasterized=True,alpha=0.4)
                plt.title('mola target')
                #plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.subplot(2,2,3)
                plt.imshow(inputs[k,0],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                plt.imshow(output[k,0],cmap='gray_r',vmin=0,vmax=1,alpha=0.5)
                plt.title('CTX predict')
                #plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.subplot(2,2,4)
                plt.imshow(inputs[k,1],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                plt.imshow(output[k,1],cmap='gray_r',vmin=0,vmax=1,alpha=0.5)
                plt.title('mola predict')
                #plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f'{plotDir}predict_{j}_{k}.jpg',dpi=300)


            plt.close()
            #plt.figure(figsize=(5,16))
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(inputs[k,0],cmap='gray')
            plt.imshow(target[k,0],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
            plt.title('Extention target')
            plt.gca().set_aspect('equal')
            #plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(inputs[k,1],cmap='gray')
            plt.imshow(target[k,1],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
            plt.title('Compression target')
            plt.gca().set_aspect('equal')
            #plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(np.clip(inputs[k,4:].transpose([1,2,0])/2+0.5,0,1))
            plt.imshow(output[k,0],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
            plt.title('Extention predict')
            plt.gca().set_aspect('equal')
            #plt.colorbar()
            plt.subplot(2,2,4)
            image = np.clip(output_image[k].transpose([1,2,0]),0,1)
            image[:,:,:] = image.mean(axis=2)[:,:,np.newaxis]
            plt.imshow(image)
            plt.imshow(output[k,1],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
            plt.title('Compression predict')
            plt.tight_layout()
            #plt.savefig(f'{plotDir}{i}_{k}.jpg',dpi=300)
            plt.savefig(f'{plotDir}predict_{j}_{k}t.jpg',dpi=300)
            plt.imsave(f'{plotDir}{j}_{k}_ctx.jpg',inputs[k,0],cmap='gray',)
            plt.imsave(f'{plotDir}{j}_{k}_mola.jpg',inputs[k,1],cmap='gray')
            plt.imsave(f'{plotDir}{j}_{k}_tw.jpg',inputs[k,-3:].transpose(1,2,0)/2+0.5)#,cmap='bwr',vmin=0,vmax=1)
            plt.imsave(f'{plotDir}{j}_{k}_ctx_shift.jpg',inputs_shift[k,0],cmap='gray')
            plt.imsave(f'{plotDir}{j}_{k}_mola_shift.jpg',inputs_shift[k,1],cmap='gray')
            plt.imsave(f'{plotDir}{j}_{k}_tw_shift.jpg',inputs_shift[k,-3:].transpose(1,2,0)/2+0.5)

# ... (Keep all your other job blocks like 'predict_all', 'predict_all_f', etc. from your original script)
# ... (They go right here, all the way to the end of the file)