import h5py
import numpy as np

#打开文件
f = h5py.File('D:\\学习\\高光谱项目\\to_lin\\to_lin\\H5_model.h5','r')
#遍历文件中的一级组
for group in f.keys():
    print(group)
    #根据一级组名获得其下面的组
    group_read = f[group]
    #遍历该一级组下面的子组
    for subgroup in group_read.keys():
        print(subgroup)
        #根据一级组和二级组名获取其下面的dataset
        dset_read = f[group+'/'+subgroup]
        #遍历该子组下所有的dataset
        for dset in dset_read.keys():
            #获取dataset数据
            dset1 = f[group+'/'+subgroup+'/'+dset]
            print(dset1.name)
            data = np.array(dset1)
            print(data.shape)
            x = data[...]
            y = data[...]
