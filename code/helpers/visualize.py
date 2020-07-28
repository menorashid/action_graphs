import matplotlib
import numpy as np;
matplotlib.use('Agg')
# matplotlib.use('PS') 
import matplotlib.pyplot as plt;
matplotlib.rcParams.update({'font.size': 16})
from matplotlib.backends.backend_pdf import PdfPages
import os;
from PIL import Image,ImageDraw,ImageFont;
import scipy.misc
import util;
import itertools

def writeHTML(file_name,im_paths,captions,height=200,width=200):
    f=open(file_name,'w');
    html=[];
    f.write('<!DOCTYPE html>\n');
    f.write('<html><body>\n');
    f.write('<table>\n');
    for row in range(len(im_paths)):
        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td>');
            f.write(captions[row][col]);
            f.write('</td>');
            f.write('    ');
        f.write('\n</tr>\n');

        f.write('<tr>\n');
        for col in range(len(im_paths[row])):
            f.write('<td><img src="');
            f.write(im_paths[row][col]);
            # f.write('" height=100%; width=100%;"/></td>');
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>');
            f.write('    ');
        f.write('\n</tr>\n');
        f.write('<p></p>');
    f.write('</table>\n');
    f.close();

def getHeatMap(arr,max_val=255):
    # cmap = plt.get_cmap('jet')
    import matplotlib as mpl

    norm = mpl.colors.Normalize(vmin=np.min(arr), vmax=np.max(arr))
    rgb_img = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues).to_rgba(arr)
    rgb_img = rgb_img[:,:,:3]
    # print rgb_img.shape
    # axes[2].axvline(g2, color=cm.ScalarMappable(norm=norm, cmap=cm.jet).to_rgba(g2))



    # print cmap
    # rgba_img = cmap(arr)
    # rgb_img = np.delete(rgba_img, 3, 2)
    # rgb_img = rgb_img*max_val
    # print rgb_img.shape,rgba_img.shape
    return rgb_img

def fuseAndSave(img,heatmap,alpha,out_file_curr=None):
    im=(img*alpha)+(heatmap*(1-alpha));

    if out_file_curr is None:
        return im;
    else:
        scipy.misc.imsave(out_file_curr,im);


def visualizeFlo(flo,file_name_x,file_name_y):
    plt.figure();plt.imshow(flo[:,:,0]);plt.savefig(file_name_x);
    plt.close();
    plt.figure();plt.imshow(flo[:,:,1]);plt.savefig(file_name_y);
    plt.close();

def createScatterOfDiffsAndDistances(diffs,title,xlabel,ylabel,out_file,dists=None):
    plt.figure();
    
    

    print out_file

    diffs_all=diffs.ravel();
    dists_all=[];
    if dists is None:
        dists_all=np.arange(diffs.shape[1]);
        dists_all=np.repeat(dists_all,diffs.shape[0]);
        # for idx in range(len(diffs)):
        #     dists_all.extend(range(1,len(diffs[idx])));
    else:
        dists_all=np.ravel(dists)
        # for dist in dists:
        #     dists_all.extend(dist);
    # bins=(max(diffs_all)-min(diffs_all),max(diffs_all)-min(diffs_all));
    heatmap, xedges, yedges = np.histogram2d(dists_all,diffs_all,bins=(100,45))
    heatmap=heatmap.T
    print heatmap.shape
    # print bins

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    plt.imshow(heatmap)
    plt.savefig(out_file);
    plt.close();


def plotMultiHist(out_file, vals, num_bins, title='',xlabel='',ylabel='',legend_entries=None, loc=0,outside=False,logscale=False,colors=None,xticks=None,ylim=None, align = 'right', density = False):

    plt.title(title);
    plt.grid(1);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    if logscale:
        plt.gca().set_xscale('log')
    # assert len(xs)==len(ys)
    alpha = 1/float(len(vals))
    handles = []
    for idx_val,val in enumerate(vals):
        if legend_entries is not None:
            handle = plt.hist(val, num_bins[idx_val], alpha=alpha, label=legend_entries[idx_val], align = align, density = density)
            # handles.append(handle)
        else:
            handle = plt.hist(val, num_bins[idx_val], alpha=alpha, align = align, density = density)
        handles.append(handle)

    if legend_entries is not None:
        if outside:
            lgd=plt.legend(loc=loc,bbox_to_anchor=(1.05, 1),borderaxespad=0.)
        else:
            lgd=plt.legend(loc=loc)    


    if xticks is not None:
        ax = plt.gca()
        ax.set_xticks(num_bins[0])
        if len(xticks)>13:
            ax.set_xticklabels(xticks, fontsize = 'small')
        else:
            ax.set_xticklabels(xticks,rotation=0)
        

    if ylim is not None:
        plt.ylim([ylim[0],ylim[1]]);

    if legend_entries is not None:
        plt.savefig(out_file,bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(out_file);

    plt.close();



def saveMatAsImage(mat,out_file,title = ''):
    fig = plt.figure()  
    plt.title(title)
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, interpolation='nearest')
    fig.colorbar(cax)


    # print list(np.where(mat[0,:]))
    # plt.imshow(mat)

    # cax = plt.matshow(mat)
    #     # ,interpolation = 'nearest');
    # fig.colorbar(cax)  
    plt.savefig(out_file);
    # fig.savefig(out_file, format='png')
    # , dpi=fig.dpi)
    # scipy.misc.imsave(out_file,mat)
    plt.close()


def createImageAndCaptionGrid(img_paths,gt_labels,indices,text_labels):
    im_paths=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    captions=[[[] for i in range(indices.shape[1]+1)] for j in range(indices.shape[0])]
    for r in range(indices.shape[0]):
        im_paths[r][0]=img_paths[r];
        captions[r][0]='GT class \n'+text_labels[gt_labels[r]]+' '+str(gt_labels[r]);
        for c in range(indices.shape[1]):
            pred_idx=indices[r][c]
            im_paths[r][c+1]=img_paths[pred_idx];
            if gt_labels[pred_idx] !=gt_labels[r]:
                captions[r][c+1]='wrong \n'+text_labels[gt_labels[pred_idx]]+' '+str(gt_labels[pred_idx]);
            else:
                captions[r][c+1]='';
    return im_paths,captions


def plotDistanceHistograms(diffs_curr,degree,out_file,title='',xlabel='Distance Rank',ylabel='Frequency',delta=0,dists_curr=None,bins=10,normed=False):
    
    if dists_curr is None:
        dists_curr=np.array(range(1,diffs_curr.shape[1]+1));
        dists_curr=np.expand_dims(dists_curr,0);
        dists_curr=np.repeat(dists_curr,diffs_curr.shape[0],0);

    # diffs_to_analyze=[0,45,90,135,180];
    # plt.ion();
    # for diff_to_analyze in diffs_to_analyze:
    diffs=diffs_curr-degree;
    diffs=abs(diffs);
    idx=np.where(diffs<=delta)
    dists=dists_curr[idx[0],idx[1]];

    plt.figure();
    print  'len(dists)',len(dists);
    plt.hist(dists,bins,normed=normed);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.savefig(out_file);
    plt.close();

def hist(dists,out_file,bins=10,normed=True,xlabel='Value',ylabel='Frequency',title='',cumulative=False):
    plt.figure();
    plt.hist(dists,bins,normed=normed,cumulative=cumulative);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.savefig(out_file);
    plt.close();    

def plotErrorBars(dict_to_plot,x_lim,y_lim,xlabel,y_label,title,out_file,margin=[0.05,0.05],loc=2):
    
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(y_label);
    
    if y_lim is None:
        y_lim=[1*float('Inf'),-1*float('Inf')];
    
    max_val_seen_y=y_lim[1]-margin[1];
    min_val_seen_y=y_lim[0]+margin[1];
    print min_val_seen_y,max_val_seen_y
    max_val_seen_x=x_lim[1]-margin[0];
    min_val_seen_x=x_lim[0]+margin[0];
    handles=[];
    for k in dict_to_plot:
        means,stds,x_vals=dict_to_plot[k];
        
        min_val_seen_y=min(min(np.array(means)-np.array(stds)),min_val_seen_y);
        max_val_seen_y=max(max(np.array(means)+np.array(stds)),max_val_seen_y);
        
        min_val_seen_x=min(min(x_vals),min_val_seen_x);
        max_val_seen_x=max(max(x_vals),max_val_seen_x);
        
        handle=plt.errorbar(x_vals,means,yerr=stds);
        handles.append(handle);
        print max_val_seen_y
    plt.xlim([min_val_seen_x-margin[0],max_val_seen_x+margin[0]]);
    plt.ylim([min_val_seen_y-margin[1],max_val_seen_y+margin[1]]);
    plt.legend(handles, dict_to_plot.keys(),loc=loc)
    plt.savefig(out_file);
    plt.close();

def plotSimple(xAndYs,out_file=None,title='',xlabel='',ylabel='',legend_entries=None,loc=0,outside=False,logscale=False,colors=None,xticks=None,ylim=None,noline = False):
    plt.title(title);
    plt.grid(1);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    if logscale:
        plt.gca().set_xscale('log')
    # assert len(xs)==len(ys)
    handles=[];
    for idx_x_y,(x,y) in enumerate(xAndYs):
        if colors is not None:
            color_curr=colors[idx_x_y];
            handle,=plt.plot(x,y,color_curr)
                # ,linewidth=2.0);
        else:
            if noline:
                handle,=plt.plot(x,y, marker = '.',linewidth = 0)
            else:
                handle,=plt.plot(x,y)
                # ,linewidth=2.0);

        handles.append(handle);
    if legend_entries is not None:
        if outside:
            lgd=plt.legend(handles,legend_entries,loc=loc,bbox_to_anchor=(1.05, 1),borderaxespad=0.)
        else:
            lgd=plt.legend(handles,legend_entries,loc=loc)    

    if xticks is not None:
        ax = plt.gca()
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1],rotation=0)

    if ylim is not None:
        plt.ylim([ylim[0],ylim[1]]);

    if out_file is not None:
        if legend_entries is not None:
            plt.savefig(out_file,bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            plt.savefig(out_file);

        plt.close();    

def writeHTMLForFolder(path_to_im,ext='jpg',height=300,width=300):
    im_files=[file_curr for file_curr in os.listdir(path_to_im) if file_curr.endswith(ext)];
    im_files.sort();
    im_paths=[[im_file_curr] for im_file_curr in im_files];
    captions=im_paths;
    out_file_html=os.path.join(path_to_im,path_to_im[path_to_im.rindex('/')+1:]+'.html');
    writeHTML(out_file_html,im_paths,captions,height=height,width=width);


def plotBars(out_file,x_vals,widths,y_val,color,xlabel='',ylabel='',title='',xlim = None, ylim = None):
    plt.figure();
    # plt.grid(1);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    
    plt.gca().bar(x_vals,y_val,widths,color=color,align = 'edge')

    if ylim is not None:
        plt.ylim(ylim )

    if xlim is not None:
        plt.xlim(xlim )
    
    plt.savefig(out_file,bbox_inches='tight');
    plt.close(); 
    # plt.bar([p + width*pos_idx for p in pos],dict_vals[legend_val],width,color=colors[pos_idx],label=legend_val)
    # dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel='',title='',width=0.25,ylim=None,loc=None):

def plotBarsSubplot(out_file,x_vals_all,widths_all,y_vals,colors,xlabel='',ylabel='',title='',xlim = None):
    # Two subplots, the axes array is 1-d
    f, ax_arr = plt.subplots(len(x_vals_all), sharex=True)
    for idx_ax_curr,ax_curr in enumerate(ax_arr):
        ax_curr.bar(x_vals_all[idx_ax_curr],y_vals[idx_ax_curr],widths_all[idx_ax_curr],color=colors[idx_ax_curr],align = 'edge')     

    if xlim is not None:
        plt.xlim(xlim )   
    plt.savefig(out_file,bbox_inches='tight');
    plt.close(); 

        # axarr[0].plot(x, y)
        # axarr[0].set_title('Sharing X axis')
        # axarr[1].scatter(x, y)

def plotGroupBar(out_file,dict_vals,xtick_labels,legend_vals,colors,xlabel='',ylabel='',title='',width=0.25,ylim=None,loc=None):
    # print 'loc',loc
    if loc is None:
        loc=2;
    # Setting the positions and width for the bars
    # if ylim is None:
    #     vals=dict_vals.values();
    #     vals=[v for v in val for val in values];
    #     ylim[
    plt.figure();
    plt.grid(1);
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    # w=len(legend_vals)*width
    # pos = np.arange(0,w*len(xtick_labels),w)
    pos=range(len(xtick_labels));

    pos = [pos_curr+(pos_curr*width) for pos_curr in pos]
    # pos[0]=0.0;
    # print pos
    # width = 0.25

    # Plotting the bars
    # fig, ax = plt.subplots(figsize=(10,5))


    # Create a bar with pre_score data,
    # in position pos,

    for pos_idx,legend_val in enumerate(legend_vals):
        # print legend_val,[p + width*pos_idx for p in pos],dict_vals[legend_val]
        # print [p + width*pos_idx for p in pos],dict_vals[legend_val],width,colors[pos_idx],legend_val
        plt.bar([p + width*pos_idx for p in pos],dict_vals[legend_val],width,color=colors[pos_idx],label=legend_val)

    ax = plt.gca()
    
    ax.set_xticks([p + len(legend_vals)/2.0 * width for p in pos])
    # print 'xticks' ,[p + len(legend_vals)/2.0 * width for p in pos]
    ax.set_xticklabels(xtick_labels,rotation=90)
    # ax.legend( legend_vals,loc=loc)
    if loc>5:
        plt.legend(legend_vals,bbox_to_anchor=(0., 0, 1., 1), ncol=2);
    else:
        ax.legend( legend_vals,loc=loc)
        # , mode="expand", borderaxespad=0.)
# Setting the x-axis and y-axis limits
    # plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim )

# Adding the legend and showing the plot
    
    # plt.gcf().subplots_adjust(top=0.9,bottom=0.15)
    # if out_file.endswith('.pdf'):
    #     with PdfPages(out_file) as pdf:
    #         pdf.savefig()
    # else:
    plt.savefig(out_file,bbox_inches='tight');
    plt.close();  

def plotBBox(img_path,bboxes,out_file,colors=None,labels=None):

    if type(img_path)==type('str'):
        im=scipy.misc.imread(img_path);
    else:
        im=img_path;

    if len(im.shape)<3:
        im=np.dstack((im,im,im));    
    im=np.asarray(im,np.uint8)
    im=Image.fromarray(im)
    
    if im.mode != "RGB":
        im.convert("RGB")
    # print im.size
    # print im.mode

    if labels is not None:
        assert len(labels)==len(bboxes);
        fontsize=max(20,im.size[0]/40);
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", fontsize)
        

    draw = ImageDraw.Draw(im)
    
    for idx_bbox,bbox in enumerate(bboxes):
        bbox_curr=[bbox[1],bbox[0],bbox[3],bbox[2]];
        # print bbox_curr
        if colors is None:
            color_curr=(255,255,255);
        elif type(colors)==type((1,2)):
            color_curr=colors;
        else:
            color_curr=colors[idx_bbox]
        # print color_curr;
        # print bbox_curr
        bbox_curr=[int(val) for val in bbox_curr]
        draw.rectangle(bbox_curr,outline=color_curr);

        if labels is not None:
            label_curr= labels[idx_bbox];
            if type(label_curr)!=type('str'):
                label_curr= '%.2f' %label_curr
            # print labels    
            color_curr= (color_curr[0],color_curr[1],color_curr[2],255);
            draw.text((bbox_curr[0],bbox_curr[1]), label_curr, font=font,fill=color_curr)

    im.save(out_file);


def writeHTMLForDifferentFolders(out_file_html,folders,captions,img_names,rel_path_replace=None,height=200,width=200):
    if rel_path_replace is None:
        string_curr=folders[0];
        rel_path_replace=string_curr[:string_curr[1:].index('/')+1];

    # print rel_path_replace

    img_paths=[];
    captions_html=[];
    
    for img_name in img_names:
        captions_row = [];
        img_paths_row = [];
        for caption_curr,folder in zip(captions,folders):
            img_paths_row.append(util.getRelPath(os.path.join(folder,img_name),rel_path_replace));
            captions_row.append(caption_curr);
        img_paths.append(img_paths_row);
        captions_html.append(captions_row);


    writeHTML(out_file_html,img_paths,captions_html,height=height,width=width);

def plotImageAndAnno(im_file,out_file,anno,color_curr=(0,255,0)):
    im=scipy.misc.imread(im_file);
    im=Image.fromarray(im)
    if im.mode != "RGB":
        im.convert("RGB")
    draw = ImageDraw.Draw(im)
    anno=list(anno.ravel());
    draw.point(anno,fill=color_curr)
    im.save(out_file);


def plot_colored_mats(out_file, mat_curr, min_val, max_val, title='', cmap=plt.cm.Blues):

    plt.imshow(mat_curr, interpolation='nearest', cmap=cmap)
    plt.clim(min_val,max_val) 
    plt.title(title)
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(out_file);
    plt.close();    

def plot_confusion_matrix(cm, classes, out_file,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_file);
    plt.close();

def main():
    print 'hello';

if __name__=='__main__':
    main();