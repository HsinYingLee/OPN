import scipy.misc
import caffe
import numpy as np
import os.path as osp 
import sys
import scipy.io as sio 
from random import shuffle
import random
from threading import Thread
from PIL import Image


class train_temporal_four(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.mean = np.array(params['mean'])
        self.batchsize = params['batchsize']
        self.trainval = params['trainval']
        self.inputtype = params['inputtype'] # 1:patch 2:upsample patch 3: whole I
    
        self.imagesize = 80
        if self.trainval == 'trainval':
            datafile = '/home/hylee/data/UCF101/UCF_trainval.mat'
        elif self.trainval == 'train':
            datafile = '/home/hylee/data/UCF101/UCF_train.mat'
        elif self.trainval == 'val':
            datafile = '/home/hylee/data/UCF101/UCF_val.mat'
        else:
            sys.stderr.write('Wrong Trainval Param!!')
            raw_input()
  
        dataset= sio.loadmat(datafile)
        files = dataset['filename']
        frame = dataset['frame']
        crop = dataset['crop']
        tuplenum = files.shape[0]
        self.tuplenum = tuplenum
        self.filelist = [None] * tuplenum
        self.framelist = frame
        self.croplist = crop
        self.randlist = range(tuplenum)
        shuffle(self.randlist)
        self.idxcounter = 0 
        for i in range(tuplenum):
            self.filelist[i] = files[i][0][0]
        self.channels = 3 
        self.height = self.imagesize
        self.width = self.imagesize

        self.top_names = ['im1', 'im2', 'im3', 'im4', 'label']
        for top_index, name in enumerate(self.top_names):
            if name == 'label':
                shape = (self.batchsize,)
            else:
                shape = (self.batchsize, self.channels, self.height, self.width)
            top[top_index].reshape(*shape)
        self.im1 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        self.im2 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        self.im3 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        self.im4 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        self.label = np.zeros((self.batchsize,))

    def reshape(self, bottom, top):
        pass
    def forward(self, bottom, top):
        cnt = 0
        tmpdata1 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        tmpdata2 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        tmpdata3 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        tmpdata4 = np.zeros((self.batchsize, self.channels, self.height, self.width))
        while cnt is not self.batchsize:

            frame_dir = self.filelist[ self.randlist[self.idxcounter] ] + '/'
            frame = [None]*5
            frame[0] = self.framelist[ self.randlist[self.idxcounter] ][0]
            frame[1] = self.framelist[ self.randlist[self.idxcounter] ][1]
            frame[2] = self.framelist[ self.randlist[self.idxcounter] ][2]
            frame[3] = self.framelist[ self.randlist[self.idxcounter] ][3]
            xcrop = self.croplist[ self.randlist[self.idxcounter],: ][0]
            ycrop = self.croplist[ self.randlist[self.idxcounter],: ][1]
            
            mirror = random.randint(0,1)

            fname1 = '/home/hylee/data/' + frame_dir + "frame%06d.jpg"%(frame[ select[0] ])
            fname2 = '/home/hylee/data/' + frame_dir + "frame%06d.jpg"%(frame[ select[1] ])
            fname3 = '/home/hylee/data/' + frame_dir + "frame%06d.jpg"%(frame[ select[2] ])
            fname4 = '/home/hylee/data/' + frame_dir + "frame%06d.jpg"%(frame[ select[3] ])
            
            img1 = Image.open(fname1)
            img2 = Image.open(fname2)
            img3 = Image.open(fname3)
            img4 = Image.open(fname4)

            ## Crop selected region out of whole images
            img1 = img1.crop((ycrop,xcrop,ycrop+self.imagesize+20,xcrop+self.imagesize+20))
            img2 = img2.crop((ycrop,xcrop,ycrop+self.imagesize+20,xcrop+self.imagesize+20))
            img3 = img3.crop((ycrop,xcrop,ycrop+self.imagesize+20,xcrop+self.imagesize+20))
            img4 = img4.crop((ycrop,xcrop,ycrop+self.imagesize+20,xcrop+self.imagesize+20))
            
            ## Mirror
            if mirror == 1:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
                img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
                img4 = img4.transpose(Image.FLIP_LEFT_RIGHT)
            
            ## Spatial Jittering
            sjx = self.imagesize
            sjy = self.imagesize
            startx = random.randint(0, img1.size[0]-sjx)
            starty = random.randint(0, img1.size[1]-sjy)
            endx = startx + sjx
            endy = starty + sjy

            sjdis = 5
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop1 = img1.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop2 = img2.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop3 = img3.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop4 = img4.crop((newx,newy,newx+sjx,newy+sjy))
        
            im1 = np.array(imgcrop1, dtype=np.float32)
            im2 = np.array(imgcrop2, dtype=np.float32)
            im3 = np.array(imgcrop3, dtype=np.float32)
            im4 = np.array(imgcrop4, dtype=np.float32)
                
            ## Channel Splitting
            rgb = random.randint(0,2)
            im1 = im1[:,:,rgb]
            rgb = random.randint(0,2)
            im2 = im2[:,:,rgb]
            rgb = random.randint(0,2)
            im3 = im3[:,:,rgb]
            rgb = random.randint(0,2)
            im4 = im4[:,:,rgb]
            im1 = np.stack((im1,)*3, axis=2)
            im2 = np.stack((im2,)*3, axis=2)
            im3 = np.stack((im3,)*3, axis=2)
            im4 = np.stack((im4,)*3, axis=2)
            im1 -= 96.5 
            im2 -= 96.5 
            im3 -= 96.5 
            im4 -= 96.5
                
            im1 = im1[:,:,::-1]
            im2 = im2[:,:,::-1]
            im3 = im3[:,:,::-1]
            im4 = im4[:,:,::-1]
            im = [None] * 4
            im[0] = im1.transpose((2,0,1))
            im[1] = im2.transpose((2,0,1))
            im[2] = im3.transpose((2,0,1))
            im[3] = im4.transpose((2,0,1))
            
            order = random.randint(0,11)
            rev = random.randint(0,1)
            ordertype = [[1,2,3,4],[1,3,2,4],[1,3,4,2],[1,2,4,3],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[3,1,2,4],[3,1,4,2],[3,2,1,4]]    
            if rev == 0:
                tmpdata1[cnt][:] = im[ordertype[order][0]-1]
                tmpdata2[cnt][:] = im[ordertype[order][1]-1]
                tmpdata3[cnt][:] = im[ordertype[order][2]-1]
                tmpdata4[cnt][:] = im[ordertype[order][3]-1]
            else: 
                tmpdata1[cnt][:] = im[ordertype[order][3]-1]
                tmpdata2[cnt][:] = im[ordertype[order][2]-1]
                tmpdata3[cnt][:] = im[ordertype[order][1]-1]
                tmpdata4[cnt][:] = im[ordertype[order][0]-1]
            
            self.label[cnt] =  order

            self.idxcounter = self.idxcounter + 1;
            if self.idxcounter == self.tuplenum:
                self.idxcounter = 0
                shuffle(self.randlist)
            cnt = cnt + 1

        top[0].data[...] = tmpdata1
        top[1].data[...] = tmpdata2
        top[2].data[...] = tmpdata3
        top[3].data[...] = tmpdata4
        top[4].data[...] = self.label
        
