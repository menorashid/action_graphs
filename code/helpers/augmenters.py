import numpy as np
import scipy.misc
import cv2
import random

def crop_center(img,cropx,cropy):
    # y,x = img.shape
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def random_crop(img,crop_size):
    assert img.shape[0]==img.shape[1]
    img_size = img.shape[0]
    assert crop_size<=img_size
    min_val = 0
    max_val = img_size-crop_size
    x = np.random.randint(min_val,max_val)
    y = np.random.randint(min_val,max_val)
    return img[y:y+crop_size,x:x+crop_size,:]

def horizontal_flip(im):
    if np.random.random()<0.5:
        im = np.array(im[:,::-1,:])
    return im

def hide_and_seek(im, div_sizes = [9,7,5,3], hide_prob = 0.5,fill_val = 0):

    # get width and height of the image
    s = im.shape
    wd = s[0]
    ht = s[1]

    grid_sizes = [0]+[im.shape[0]//div for div in div_sizes]
    # print grid_sizes
    # possible grid size, 0 means no hiding
    # grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_probs = [0.1,0.2,0.3,0.4,0.5]
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]
    hide_prob = hide_probs[random.randint(0,len(hide_probs)-1)]
    
    # hide the patches
    if grid_size!=0:
        for x in range(0,wd,grid_size):
            for y in range(0,ht,grid_size):
                x_end = min(wd, x+grid_size)  
                y_end = min(ht, y+grid_size)
                if(random.random() <=  hide_prob):
                    im[x:x_end,y:y_end]=fill_val

    return im




def augment_image( im, list_of_to_dos = ['flip','rotate','scale_translate'],mean_im=None, std_im=None, im_size = 96, color = False):
        # khorrami augmentation for ck+
        # trying to get baseline results 

        # a. Flip: The image is horizontally mirrored with probability 0.5.
        # b. Rotation: A random theta is sampled uniformly from the range [-5, 5] degrees and the image is rotated by theta.
        # c. Scale: A random alpha is sampled uniformly from the range [0.7, 1.4] and the image is scaled by alpha.
        # d. Translation: A random [x, y] vector is sampled and the image is translated by [x, y]. x and y are defined such that:
        # x ~ Uniform(-delta/2, delta/2)
        # y ~ Uniform(-delta/2, delta/2)
        # where delta = (alpha-1)*96.
        # e. Intensity Change: The pixels of an image (p(i, j)) are changed using the following formula: 
        # p*(i, j) = (p(i, j)^a) * b + c 
        # where a, b, and c are defined as:
        # a ~ Uniform(0.25, 4)
        # b ~ Uniform(0.7, 1.4)
        # c ~ Uniform(-0.1, 0.1)
        
        rot_range =[-5,5]
        alpha_range = [0.7,1.4]
        a_range = [0.25,4]
        b_range = [0.7, 1.4]
        c_range = [-0.1,0.1]
        

        if 'pixel_augment' in list_of_to_dos:
            assert color==False

            im = im[:,:,0]
            im = im*std_im
            im = np.clip(im + mean_im,0,255)
            im = im/255.
            

            a_b_c = np.random.random_sample((3,))
            a = a_b_c[0]*(a_range[1]-a_range[0]) + a_range[0]
            b = a_b_c[1]*(b_range[1]-b_range[0]) + b_range[0]
            c = a_b_c[2]*(c_range[1]-c_range[0]) + c_range[0]
            im = (im**a)*b + c
            im = im*255.
            im = (im - mean_im)/std_im
            im = im[:,:,np.newaxis]
        
        if not color:
            im = np.concatenate((im,im,im),2)

        min_im = np.array([np.min(im[:,:,idx_arr]) for idx_arr in range(im.shape[2])])
        min_im = min_im[np.newaxis,np.newaxis,:]

        max_im = np.array([np.max(im[:,:,idx_arr]) for idx_arr in range(im.shape[2])])
        max_im = max_im[np.newaxis,np.newaxis,:]
        # print max_im.shape
        im = im-min_im
        # max_im = np.max(im,2,keepdims=True)
        # print max_im
        im = im/max_im 
        # print np.min(im), np.max(im)
        # print im.shape
        # print 'HELLO'
        # raw_input()
        

        # flip it
        if 'flip' in list_of_to_dos:
            if np.random.random()<0.5:
                im = im[:,::-1,:]

        if 'rotate' in list_of_to_dos:
            deg = np.random.random()*(rot_range[1]-rot_range[0]) + rot_range[0]
            im = scipy.misc.imrotate(im,deg)

        if 'scale_translate' in list_of_to_dos:
            alpha = np.random.random()*(alpha_range[1]-alpha_range[0]) + alpha_range[0]
            delta = abs(alpha-1)*im_size
            delta_range = [-1*delta/2.,delta/2.]
            assert delta_range[0]<=delta_range[1]

            im_rs = scipy.misc.imresize(im, alpha)
            
            vec_translate = np.random.random_sample((2,))*(delta_range[1]-delta_range[0]) + delta_range[0]
            vec_translate = np.around(vec_translate).astype(dtype = np.int)
            padding = [0,0,0,0]
            for dim_num in range(2):
                if vec_translate[dim_num]<0:
                    padding[2+dim_num] = -vec_translate[dim_num]
                else:
                    padding[dim_num] = vec_translate[dim_num]

            im_rs = np.pad(im_rs,((padding[0],padding[1]),(padding[2],padding[3]),(0,0)),'constant')
            
            start_idx_im = [0,0]
            end_idx_im = [0,0]
            start_idx_im_rs = [0,0]
            end_idx_im_rs = [0,0]

            for dim_num in range(2):
                if im_rs.shape[dim_num]<im_size:
                    start_idx_im_rs[dim_num] = 0
                    end_idx_im_rs[dim_num] = im_rs.shape[dim_num]
                    start_idx_im[dim_num] = max(int(round(im_size/float(2) -im_rs.shape[dim_num]/float(2))),0)
                else:
                    start_idx_im_rs[dim_num] = max(int(round(im_rs.shape[dim_num]/float(2) - im_size/float(2) )),0)
                    end_idx_im_rs[dim_num] = min(start_idx_im_rs[dim_num]+im_size,im_rs.shape[dim_num])

                    start_idx_im[dim_num] = 0
                end_idx_im[dim_num] = min(start_idx_im[dim_num] + (end_idx_im_rs[dim_num]-start_idx_im_rs[dim_num]),im_size)


            im = np.zeros(im.shape)
            
            im[start_idx_im[0]:end_idx_im[0],start_idx_im[1]:end_idx_im[1],:] = im_rs[start_idx_im_rs[0]:end_idx_im_rs[0],start_idx_im_rs[1]:end_idx_im_rs[1],:]

        # print color, im.shape
        if not color:
            im = im[:,:,:1]
            max_im = max_im.squeeze()[0]
            min_im = min_im.squeeze()[0]
            # print max_im, min_im

        if 'rotate' in list_of_to_dos or 'scale_translate' in list_of_to_dos:
            im = im/255.

        # print color, im.shape
        im = im * max_im
        im = im + min_im
        # print np.min(im), np.max(im), im.shape
        # scipy.misc.imsave('../scratch/hello.jpg',im)

        # raw_input()
        return im