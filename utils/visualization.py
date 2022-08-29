from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import copy

LANDCOVER_CLASSES = {
    0: "Clouds",
    62: "Artificial surfaces and constructions",
    73: "Cultivated areas",
    75: "Vineyards",
    82: "Broadleaf tree cover",
    83: "Coniferous tree cover",
    102: "Herbaceous vegetation",
    103: "Moors and Heathland",
    104: "Sclerophyllous vegetation",
    105: "Marshes",
    106: "Peatbogs",
    121: "Natural material surfaces", 
    123: "Permanent snow covered surfaces",
    162: "Water bodies",
    255: "No data",
}

LC_CONVERTED = {k: (i,LANDCOVER_CLASSES[k]) for i, k in enumerate(sorted(list(LANDCOVER_CLASSES.keys())))}
LC_CONVERTED_CLASSES = {v[0]: v[1] for k, v in LC_CONVERTED.items()}
lc_convert = np.vectorize(lambda x: LC_CONVERTED[x][0])

COLORS = np.array(
    [[255,255,255],
    [210,0,0],
    [253,211,39],
    [176,91,16],
    [35,152,0],
    [8,98,0],
    [249,150,39],
    [141,139,0],
    [95,53,6],
    [149,107,196],
    [77,37,106],
    [154,154,154],
    [106,255,255],
    [20,69,249],
    [255,255,255]]
)

def colorize(data, colormap = "ndvi", mask_red = None, mask_blue = None):
    t,h,w = data.shape
    in_data = data.reshape(-1)
    if mask_red is not None:
        in_data = np.ma.array(in_data, mask = mask_red.reshape(-1))  

    cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256) if colormap == "ndvi" else copy.copy(plt.get_cmap(colormap))
    cmap.set_bad(color='red')

    if mask_blue is None:
        return cmap(in_data)[:,:3].reshape((t,h,w,3))
    else:
        out = cmap(in_data)[:,:3].reshape((t,h,w,3))
        return np.stack([np.where(mask_blue, out[:,:,:,0],np.zeros_like(out[:,:,:,0])), 
                            np.where(mask_blue, out[:,:,:,1],np.zeros_like(out[:,:,:,1])), 
                            np.where(mask_blue, out[:,:,:,2],0.1*np.ones_like(out[:,:,:,2]))], axis = -1)


def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    padded = np.zeros((nindex, height + 2, width + 2, intensity))
    padded[:,1:-1,1:-1,:] = array
    nindex, height, width, intensity = padded.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    result = (padded.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def cube_gallery(cube, variable = "rgb", vegetation_mask = None, cloud_mask = True, save_path = None):
    """
    Plots a gallery view from a given Cube.
    Args:
        cube (np.ndarray): Numpy Array or loaded NPZ of Cube or path to Cube.
        variable (str, optional):  One of "rgb", "ndvi", "rr","pp","tg","tn","tx". Defaults to "rgb".
        vegetation_mask (np.ndarray, optional): If given uses this as red mask over non-vegetation. S2GLC data. Defaults to None.
        cloud_mask (bool, optional): If True tries to use the last channel from the cubes sat imgs as blue cloud mask, 1 where no clouds, 0 where there are clouds. Defaults to True.
        save_path (str, optional): If given, saves PNG to this path. Defaults to None.
    Returns:
        plt.Figure: Matplotlib Figure
    """    

    assert(variable in ["rgb", "ndvi", "rr","pp","tg","tn","tx"])

    if isinstance(cube, str) or isinstance(cube, Path):
        cube = np.load(cube)

    if isinstance(cube, np.lib.npyio.NpzFile):
        if variable in ["rgb","ndvi"]:
            if "highresdynamic" in cube:
                data = cube["highresdynamic"]
            else:
                for k in cube:
                    if 128 in cube[k].shape:
                        data = cube[k]
                        break
                raise ValueError("data does not contain satellite imagery.")
        elif variable in ["rr","pp","tg","tn","tx"]:
            if "mesodynamic" in cube:
                data = cube["mesodynamic"]
            else:
                raise ValueError("data does not contain E-OBS.")
    elif isinstance(cube, np.ndarray):
        data = cube

    hw = 128 if variable in ["rgb","ndvi"] else 80
    hw_idxs = [i for i,j in enumerate(data.shape) if j == hw]
    assert(len(hw_idxs) > 1)
    if len(hw_idxs) == 2 and hw_idxs != [1,2]:
        c_idx = [i for i,j in enumerate(data.shape) if j == min([j for j in data.shape if j != hw])][0]
        t_idx = [i for i,j in enumerate(data.shape) if j == max([j for j in data.shape if j != hw])][0]
        data = np.transpose(data,(t_idx,hw_idxs[0],hw_idxs[1],c_idx))

    if variable == "rgb":
        targ = np.stack([data[:,:,:,2],data[:,:,:,1],data[:,:,:,0]], axis = -1)
        targ[targ<0] = 0
        targ[targ>0.5] = 0.5
        targ = 2*targ
        if data.shape[-1] > 4 and cloud_mask:
            mask = data[:,:,:,-1]
            zeros = np.zeros_like(targ)
            zeros[:,:,:,2] = 0.1
            targ = np.where(np.stack([mask]*3,-1).astype(np.uint8) | np.isnan(targ).astype(np.uint8), zeros, targ)
        else:
            targ[np.isnan(targ)] = 0

    elif variable == "ndvi":
        if data.shape[-1] == 1:
            targ = data[:,:,:,0]
        else:
            targ = (data[:,:,:,3] - data[:,:,:,2]) / (data[:,:,:,2] + data[:,:,:,3] + 1e-6)
        if data.shape[-1] > 4 and cloud_mask:
            cld_mask = 1 - data[:,:,:,-1]
        else:
            cld_mask = None
        
        if vegetation_mask is not None:
            if isinstance(vegetation_mask, str) or isinstance(vegetation_mask, Path):
                vegetation_mask = np.load(vegetation_mask)
            if isinstance(vegetation_mask, np.lib.npyio.NpzFile):
                vegetation_mask = vegetation_mask["landcover"]
            vegetation_mask = vegetation_mask.reshape(hw, hw)
            lc_mask = 1 - (vegetation_mask > 63) & (vegetation_mask < 105)
            lc_mask = np.repeat(lc_mask[np.newaxis,:,:],targ.shape[0], axis = 0)
        else:
            lc_mask = None
        targ = colorize(targ, colormap = "ndvi", mask_red = lc_mask, mask_blue = cld_mask)
    
    elif variable == "rr":
        targ = data[:,:,:,0]
        targ = colorize(targ, colormap = 'Blues', mask_red = np.isnan(targ))
    elif variable == "pp":
        targ = data[:,:,:,1]
        targ = colorize(targ, colormap = 'rainbow', mask_red = np.isnan(targ))
    elif variable in ["tg","tn","tx"]:
        targ = data[:,:,:, 2 if variable == "tg" else 3 if variable == "tn" else 4]
        targ = colorize(targ, colormap = 'coolwarm', mask_red = np.isnan(targ))

    grid = gallery(targ)

    fig = plt.figure(dpi = 300)
    plt.imshow(grid)
    plt.axis('off')
    if variable != "rgb":
        colormap = {"ndvi": "ndvi", "rr": "Blues", "pp": "rainbow", "tg": "coolwarm", "tn": "coolwarm", "tx": "coolwarm"}[variable]
        cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256) if colormap == "ndvi" else copy.copy(plt.get_cmap(colormap))
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        vmin, vmax = {"ndvi": (0,1), "rr": (0,50), "pp": (900,1100), "tg": (-50,50), "tn": (-50,50), "tx": (-50,50)}[variable]
        label = {"ndvi": "NDVI", "rr": "Precipitation in mm/d", "pp": "Sea-level pressure in hPa", "tg": "Mean temperature in °C", "tn": "Minimum Temperature in °C", "tx": "Maximum Temperature in °C"}[variable]
        plt.colorbar(cm.ScalarMappable(norm = clr.Normalize(vmin = vmin, vmax = vmax), cmap = cmap), cax = cax, label = label)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parents[0].mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches='tight', transparent=True)

    return fig


def cube_ndvi_timeseries(pred, targ, vegetation_mask = None, save_path = None):
    """
    Plots a timeseries view of a predicted cube vs its respective target.
    Args:
        pred (str, Path, np.lib.npyio.NpzFile, np.ndarray): Cube with prediction
        targ (str, Path, np.lib.npyio.NpzFile, np.ndarray): Cube with target
        vegetation_mask (str, Path, np.lib.npyio.NpzFile, np.ndarray, optional): Cube with S2GLC Landcover mask. Defaults to None.
        save_path (str, optional): If given, saves PNG to this path. Defaults to None.
    Returns:
        plt.Figure: Matplotlib Figure
    """    
    if isinstance(pred, str) or isinstance(pred, Path):
        pred_cube = np.load(pred)
        pred_ndvi = pred_cube["highresdynamic"].astype(np.float32)
    elif isinstance(pred, np.lib.npyio.NpzFile):
        pred_cube = pred
        pred_ndvi = pred_cube["highresdynamic"].astype(np.float32)
    else:
        assert(isinstance(pred, np.ndarray))
        pred_ndvi = pred_ndvi
    
    if pred_ndvi.shape[-2] > 1:
        pred_ndvi = (pred_ndvi[:,:,3,:] - pred_ndvi[:,:,2,:]) / (pred_ndvi[:,:,2,:] + pred_ndvi[:,:,3,:] + 1e-6)
    
    if isinstance(targ, str) or isinstance(targ, Path):
        targ_cube = np.load(targ)
        targ_data = targ_cube["highresdynamic"].astype(np.float32)
    elif isinstance(targ, np.lib.npyio.NpzFile):
        targ_cube = targ
        targ_data = targ_cube["highresdynamic"].astype(np.float32)
    else:
        assert(isinstance(targ, np.ndarray))
        targ_data = targ

    if targ_data.shape[-2] == 1:
        targ_ndvi = targ_data
    else:
        targ_ndvi = (targ_data[:,:,3,:] - targ_data[:,:,2,:]) / (targ_data[:,:,2,:] + targ_data[:,:,3,:] + 1e-6)
    if targ_data.shape[-2] > 4:
        targ_ndvi = np.ma.array(targ_ndvi, mask = targ_data[:,:,-1,:].astype(bool))

    if vegetation_mask is not None:
        if isinstance(vegetation_mask, str) or isinstance(vegetation_mask, Path):
            vegetation_mask = np.load(vegetation_mask)
        if isinstance(vegetation_mask, np.lib.npyio.NpzFile):
            landcover = vegetation_mask["landcover"]
        else:
            landcover = vegetation_mask
        temp = np.concatenate([landcover.reshape((1,128,128)), np.indices((128,128))], axis = 0)
        df = pd.DataFrame(temp.reshape(3,-1).T, columns = ["lc", "x", "y"])
        coords = df[(df.lc > 63) & (df.lc < 105)].groupby('lc').agg(np.random.choice).to_numpy()
        if coords.shape[0] < 8:
            coords = np.concatenate([coords, np.indices((128,128)).reshape(2,-1).T[np.random.choice(128*128, 8-coords.shape[0])]], axis = 0)
    else:
        coords = np.indices((128,128)).reshape(2,-1).T[np.random.choice(128*128, 8)]

    fig, axs = plt.subplots(4,3, dpi = 450)
    for idx, ax in enumerate(axs.reshape(-1)): 
        if idx == 0:
            cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256)
            cmap.set_bad(color='red')
            ndvi = ax.imshow(targ_ndvi.mean(-1), cmap = cmap, vmin = 0, vmax = 1)
            ncbar = fig.colorbar(ndvi, ax=axs[0,0])
            ncbar.ax.tick_params(labelsize=6)
            ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.scatter(coords[:,0], coords[:,1], c = "grey", s = 1)
            annotations=[f"{i}" for i in range(1,9)]
            for i, label in enumerate(annotations):
                ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "orange")
            ax.set_title("Mean Target NDVI", fontsize = 6, loc = "left")
        elif idx == 1:
            dem = ax.imshow(2000 * (2*targ_cube["highresstatic"].astype(np.float32).reshape((128,128))-1), cmap = "terrain")
            dcbar = fig.colorbar(dem, ax=axs[0,1])
            dcbar.ax.tick_params(labelsize=6)
            ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.scatter(coords[:,0], coords[:,1], c = "grey", s = 1)
            annotations=[f"{i}" for i in range(1,9)]
            for i, label in enumerate(annotations):
                ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "black")
            ax.set_title("EU-DEM", fontsize = 6, loc = "left")
            #ax.tick_params(labelsize = 6)
        elif idx == 2:
            if vegetation_mask is not None:
                cmap = clr.ListedColormap(COLORS/255.)
                bounds = [i-0.5 for i in range(16)]
                norm = clr.BoundaryNorm(bounds, cmap.N)
                lac = ax.imshow(lc_convert(landcover.reshape((128,128))), cmap = cmap, norm = norm)
                lcbar = fig.colorbar(lac, ax=axs[0,2], ticks=sorted(list(LC_CONVERTED_CLASSES.keys())))
                lcbar.ax.set_yticklabels([LC_CONVERTED_CLASSES[i] for i in sorted(list(LC_CONVERTED_CLASSES.keys()))], fontsize = 3)
                lcbar.ax.tick_params(labelsize=4)
                ax.scatter(coords[:,0], coords[:,1], c = "black", s = 1)
                annotations=[f"{i}" for i in range(1,9)]
                for i, label in enumerate(annotations):
                    ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "black")
                ax.set_title("S2GLC Landcover", fontsize = 6, loc = "left")
                ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            else:
                continue
        elif idx == 3:
            ax.scatter(np.linspace(0,targ_ndvi.shape[-1],targ_ndvi.shape[-1], endpoint = False),targ_ndvi.mean((0,1)).flatten(), c = "orange", s = 0.5)
            ax.plot(np.linspace(targ_ndvi.shape[-1]-pred_ndvi.shape[-1],targ_ndvi.shape[-1],pred_ndvi.shape[-1], endpoint = False),pred_ndvi.mean((0,1)).flatten(), label = f"Pred Mean", lw = 0.5)
            ax.set_ylim(0.,1.)
            ax.set_title("NDVI mean value", fontsize = 6, loc = "left")
            ax.tick_params(labelsize = 6)
        else:
            i = idx - 4
            x, y = coords[i]
            ax.scatter(np.linspace(0,targ_ndvi.shape[-1],targ_ndvi.shape[-1], endpoint = False),targ_ndvi[x,y,:].flatten(), c = "orange", s = 0.5)
            ax.plot(np.linspace(targ_ndvi.shape[-1]-pred_ndvi.shape[-1],targ_ndvi.shape[-1],pred_ndvi.shape[-1], endpoint = False),pred_ndvi[x,y,:].flatten(), lw = 0.5)
            ax.set_ylim(0.,1.)
            if vegetation_mask is None:
                ax.set_title(f"NDVI Point {i+1}", fontsize = 6, loc = "left")
            else:
                lc = LANDCOVER_CLASSES[int(vegetation_mask["landcover"].reshape((128,128))[x,y])]
                ax.set_title(f"NDVI Point {i+1},\n{lc}", fontsize = 6, loc = "left")
            ax.tick_params(labelsize = 6)

    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    p00 = axs[0, 0].get_position()
    p01 = axs[0, 1].get_position()
    
    p10 = axs[1, 0].get_position()
    p11 = axs[1, 1].get_position()
    p12 = axs[1, 2].get_position()
    p00c = ncbar.ax.get_position()
    p00c = [(p10.x0 + (p00c.x0 - p00.x0)), p00c.y0, p00c.width, p00c.height]
    ncbar.ax.set_position(p00c)
    p01c = dcbar.ax.get_position()
    p01c = [(p11.x0 + (p01c.x0 - p01.x0)), p01c.y0, p01c.width, p01c.height]
    dcbar.ax.set_position(p01c)
    if vegetation_mask is not None:
        p02 = axs[0, 2].get_position()
        p02c = lcbar.ax.get_position()
        p02c = [(p12.x0 + (p02c.x0 - p02.x0)), p02c.y0, p02c.width, p02c.height]
        lcbar.ax.set_position(p02c)
        p02 = [p12.x0, p02.y0, p02.width, p02.height]
        axs[0, 2].set_position(p02)

    p00 = [p10.x0, p00.y0, p00.width, p00.height]
    axs[0, 0].set_position(p00)
    p01 = [p11.x0, p01.y0, p01.width, p01.height]
    axs[0, 1].set_position(p01)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parents[0].mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 450, bbox_inches='tight', transparent=True)

    return fig

def visualize_seq(file, mode, idx, fuse = False, white = False):
    file = file % (mode, idx)
    pred = np.load(file)
    if fuse:
        import torch
        #print("yes")
        m = torch.nn.Upsample(scale_factor=2, mode='nearest')
        prev = torch.tensor(pred[:,0:4,::2,::2])
        print(pred.shape)
        prev = m(prev)
        print(pred.shape)
        prev = prev.numpy()
        pred[:,0:4,:,:] = prev
    pred = np.squeeze(np.transpose(pred, (0,2,3,1)))
    length = pred.shape[2]
    #print(np.sum(pred==0))
    if length > 4:
        pred = pred.reshape((128, 128, 4, -1), order='F')
        #print(pred.shape)
        local_paddings = np.zeros(shape = (128, 128, 3, pred.shape[3]))
        #print(local_paddings.shape)
        pred_ = np.concatenate((pred, local_paddings), axis = 2)
        paddings =np.zeros(shape = (128,128,7,30 - pred.shape[3]))
        pred_ = np.concatenate((pred_, paddings), axis = 3)
    elif length == 4:
        pred = pred.reshape((128, 128, 4, -1), order='F')
        #print(pred.shape)
        local_paddings = np.zeros(shape = (128, 128, 3, pred.shape[3]))
        pred_ = np.concatenate((pred, local_paddings), axis = 2)
        paddings =np.zeros(shape = (128,128,7,30 -pred.shape[3]))
        pred_ = np.concatenate((pred_, paddings), axis = 3)
    print(pred_.shape)
    if white:
        mask = (pred_== 0)
        pred_  = pred_ + mask * 0.5
    cube_gallery(pred_,cloud_mask = True)
    #return 1 - mask[:,:,0,1]

# usable
def visualize_res(file, idx, use_mask = False, xlim = 0.97, loop = 10, denormalize = False, rgb = False,  heatmap=False, histogram = False, start=4, end = 8,threshold = 0.99, patch_only = False, patch_row = 0, patch_col = 0, patch_size = 32):
    gt_str = "results_ground_truth"
    pred_str = "results_outputs"
    input_str = "results_cloudys"
    #print(file)
    file0 = file % (pred_str, idx)
    
    pred = np.load(file0)
    pred = np.squeeze(np.transpose(pred, (0,2,3,1)))
    
    file1 = file % (gt_str, idx)
    gt = np.load(file1)
    gt = np.squeeze(np.transpose(gt, (0,2,3,1)))
    
    file2 = file % (input_str, idx)
    inp = np.load(file2)
    inp = np.squeeze(np.transpose(inp, (0,2,3,1)))
    if gt.shape[2] > 4: gt = gt[:,:,start:end]
    if pred.shape[2] > 4: pred = pred[:,:,start:end]
    if inp.shape[2] > 4: inp = inp[:,:,start:end]

    ##print(inp.shape, gt.shape, pred.shape)
    
    if patch_only:
        gt = gt[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
        pred = gt[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
        inp = inp[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
        print("cropped size,", inp.shape, gt.shape, pred.shape)
    if rgb:
        gt_r = gt[:,:,2]
        gt_g = gt[:,:,1]
        gt_b = gt[:,:,0]
        pred_r = pred[:,:,2]
        pred_g = pred[:,:,1]
        pred_b = pred[:,:,0]
        
        ax = sns.heatmap(np.abs(gt_r - pred_r), vmin = 0, vmax = 0.1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Reds')
        plt.show()
        
        ax = sns.heatmap(np.abs(gt_g - pred_g), vmin = 0, vmax = 0.1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Greens')
        plt.show()
        
        ax = sns.heatmap(np.abs(gt_b - pred_b), vmin = 0, vmax = 0.1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Blues')
        plt.show()
        
        diff = np.abs(gt-pred)[:,:,::-1]
        diff = diff[:,:,1:4]
        print("diff",diff.shape)
        plt.imshow((diff))
        plt.show()
        
    if use_mask:
        mask = (inp == 0.0)
        pred = pred * mask
        gt = gt * mask
    
#     gt[gt<0] = 0
#     gt[gt>0.5] = 0.5
#     gt = gt * 2
    
#     pred[pred<0] = 0
#     pred[pred>0.5] = 0.5
#     pred = pred * 2
    if denormalize:
        gt = gt * 255
        pred = pred * 255
    res = np.abs(gt - pred)
    #res = np.sqrt((res * 255) ** 2)
    
    res = np.sum(res, axis = 2)
    res = res/4
    #print("patch area,", patch_size ** 2)
    #print("patch loss", np.sum(res[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size])/patch_size ** 2)
    if use_mask:
        print("mask area,", np.sum(mask)/4)
        print("mask area loss", np.sum(res)/(np.sum(mask)/4))
    #return np.sum(res[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size])/patch_size ** 2
    #sim = F.cosine_similarity(input1, input2)
    dot = np.sum(gt * pred, axis = 2)
    pred_l2 = np.sqrt(np.sum(pred ** 2, axis = 2))
    gt_l2 = np.sqrt(np.sum(gt ** 2, axis = 2))
    #print("dot, pred_l2, gt_l2",dot.shape, pred_l2.shape, gt_l2.shape)
    sim = dot/(pred_l2 * gt_l2 + 1e-5)
    #print("sim", sim.shape)
    
    
    if heatmap:
        vmax = 0.1
        if denormalize: vmax = vmax * 255
        ax = sns.heatmap(res, vmin = 0, vmax = vmax, annot = False,xticklabels=False, yticklabels=False)
        plt.show()
        
        ax = sns.heatmap(sim, vmin = 0.98, vmax = 1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Greys')
        plt.show()
        
        
        sim[sim>threshold]=1
        sim[sim<threshold]=0
        ax = sns.heatmap(sim, vmin = 0, vmax = 1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Greys')
        plt.show()
        
        
    
    
    #sim = dot
    if histogram:
        #ax = sns.heatmap(sim, vmin = 0, vmax = 1, annot = False,xticklabels=False, yticklabels=False)
        #plt.show()
        
        fig, ax = plt.subplots()
        sns.histplot(sim.flatten(), bins = 10000, ax=ax)
        ax.grid(axis='y')
        ax.yaxis.grid(True) # Hide the horizontal gridlines
        ax.xaxis.grid(True)
        
        plt.xlim([xlim,1])
        plt.ylim([0,400])
        plt.xlabel("Cosine Similarity (Cloudy Only)")
        plt.axvline(x=np.median(sim[sim > 0.98]),
            color='red',ls='--')
        plt.text(np.median(sim[sim > 0.98]),200,'%0.5f' % np.median(sim[sim > 0.98]))
        plt.show()
    
    res_l = []
    for idx in range(loop):
        file0 = file % (pred_str, idx)
        pred = np.load(file0)
        pred = np.squeeze(np.transpose(pred, (0,2,3,1)))

        file1 = file % (gt_str, idx)
        gt = np.load(file1)
        gt = np.squeeze(np.transpose(gt, (0,2,3,1)))
        
        file2 = file % (input_str, idx)
        inp = np.load(file2)
        inp = np.squeeze(np.transpose(inp, (0,2,3,1)))
        
        if gt.shape[2] > 4: gt = gt[:,:,start:end]
        if pred.shape[2] > 4: pred = pred[:,:,start:end]
        if inp.shape[2] > 4: inp = inp[:,:,start:end]
        #print(inp.shape, gt.shape, mask.shape)
        if use_mask:
            mask = (inp == 0)
            if inp.shape[2] != pred.shape[2]:
                mask = mask[:,:,start:end]
            gt = gt * mask
            pred = pred * mask
        if gt.shape[2] > 4: gt = gt[:,:,start:end]
        if pred.shape[2] > 4: pred = pred[:,:,start:end]
        res = gt - pred
        res = res ** 2
        if use_mask:
            #res = np.sum(res)/(np.sum(mask)/4)
            res = np.sum(res)/(np.sum(mask))
        else:
            #res = np.mean(np.sum(res,axis=2))
            res = np.mean(res)
        res_l.append(res)
    print("avg loss", np.mean(np.array(res_l)))
    print("############")
    return res_l


def min_max_stretch(img):
    return (img - img.min()) / (img.max() - img.min())


# visualize inpainted images
def visualize_ls(file, idx, stretch=4, show_res=True, show_trip=True, vmax=100):
    gt_str = "results_ground_truth"
    pred_str = "results_outputs"
    input_str = "results_cloudys"
    sentinel_str = "results_sentinels"
    landsat_str = "results_landsats"
    #print(file)
    file0 = file % (pred_str, idx)
    
    pred = np.load(file0)
    pred = np.squeeze(np.transpose(pred, (0,2,3,1)))
#     pred = np.sqrt(pred) * stretch
#     pred = pred / 10000
    pred = np.nan_to_num(pred)
    pred = min_max_stretch(pred)
    print("pred max", pred.max(axis=(0, 1)))
    print("pred shape", pred.shape)
    
    file1 = file % (gt_str, idx)
    gt = np.load(file1)
    gt = np.squeeze(np.transpose(gt, (0,2,3,1)))
#     gt = np.sqrt(gt) * stretch
#     gt = gt / 10000
    gt = min_max_stretch(gt)
    print("gt max", gt.max(axis=(0, 1)))
    
    file2 = file % (input_str, idx)
    inp = np.load(file2)
    inp = np.squeeze(np.transpose(inp, (0,2,3,1)))
#     inp = np.sqrt(inp) * stretch
#     inp = inp / 10000
    inp = min_max_stretch(inp)
#     print("input max", inp.shape)
    
    file3 = file % (sentinel_str, idx)
    sen = np.load(file3)
    sen = np.squeeze(np.transpose(sen, (0,2,3,1)))
#     sen = np.sqrt(sen) * stretch
    print("sen max", sen.max(axis=(0, 1)))
    print("sen min", sen.min(axis=(0, 1)))
#     sen = sen / 10000
    sen = min_max_stretch(sen)
    print("sen max", sen.max(axis=(0, 1)))
    
    file4 = file % (landsat_str, idx)
    land = np.load(file4)
    land = np.squeeze(np.transpose(land, (0,2,3,1)))
#     land = np.sqrt(land) * stretch
    land = min_max_stretch(land)
#     land = land / (256 * 256)
    print("land max", land.max(axis=(0, 1)))

#     if gt.shape[2] > 4: gt = gt[:,:,start:end]
#     if pred.shape[2] > 4: pred = pred[:,:,start:end]
#     if inp.shape[2] > 4: inp = inp[:,:,start:end]

    ##print(inp.shape, gt.shape, pred.shape)
    
#     if patch_only:
#         gt = gt[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
#         pred = gt[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
#         inp = inp[patch_row:patch_row+patch_size, patch_col:patch_col+patch_size, :]
#         print("cropped size,", inp.shape, gt.shape, pred.shape)
        

#     print("Cloudy input:")
#     plt.imshow((inp).astype(np.uint8))
#     plt.show()
    
#     print("Ground truth:")
#     plt.imshow((gt).astype(np.uint8))
#     plt.show()
    
#     print("Inpainted output:")
#     plt.imshow((pred).astype(np.uint8))
#     plt.show()
    
    if show_res:
        diff = np.abs(gt-pred)
    #     diff = diff[:,:,1:4]
        print("diff",diff.shape)
        plt.axis('off')
        diff = np.sum(diff, axis = 2)
        diff = diff/3
        ax = sns.heatmap(diff, vmin = 0, vmax = vmax, annot = False, xticklabels=False, yticklabels=False)
        plt.show()
        
    if show_trip:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 5, hspace=0.1, wspace=0.1)
        axs = gs.subplots(sharex=False, sharey=False)
        fig.suptitle('Sharing both axes')
        for i in range(5):
            axs[i].autoscale()
            axs[i].axis('off')

        axs[0].imshow(inp)
        axs[1].imshow(gt)
        axs[2].imshow(pred)
        axs[3].imshow(sen)
        axs[4].imshow(land)
#         axs[4].imshow(land.astype(np.uint8))
        plt.show()
        
    return gt, pred
    

def val_metrics(file, num, use_mask=False):
    gt_str = "results_ground_truth"
    pred_str = "results_outputs"
    input_str = "results_cloudys"
    #print(file)

#     print("input shape", inp.shape)

    res_l = []
    psnr_l = []
    ssim_l = []
    sam_l = []

    for idx in range(num):
        file0 = file % (pred_str, idx)
        pred = np.load(file0)
        pred = np.squeeze(np.transpose(pred, (0,2,3,1)))
        pred = np.sqrt(pred)
        pred = np.nan_to_num(pred)
#         print("pred shape", pred.shape)

        file1 = file % (gt_str, idx)
        gt = np.load(file1)
        gt = np.squeeze(np.transpose(gt, (0,2,3,1)))
        gt = np.sqrt(gt)
        gt = np.nan_to_num(gt)
#         print("gt shape", gt.shape)

        file2 = file % (input_str, idx)
        inp = np.load(file2)
        inp = np.squeeze(np.transpose(inp, (0,2,3,1)))
        inp = np.sqrt(inp)
        
        if use_mask:
            mask = (inp == 0)
            if inp.shape[2] != pred.shape[2]:
                mask = mask[:,:,start:end]
            gt = gt * mask
            pred = pred * mask
            
        res = gt - pred
        res = res ** 2
#         print("res shape", res.shape)
        
        mse_loss = nn.MSELoss()
        #print(pred.shape, gt.shape)
        loss = mse_loss(torch.Tensor(pred).flatten(), torch.Tensor(gt).flatten())
#         print(idx, loss)
        
        #print("loss dif", loss, np.mean(res))
        if use_mask:
            #res = np.sum(res)/(np.sum(mask)/4)
            res = np.sum(res)/(np.sum(mask))
        else:
            #res = np.mean(np.sum(res,axis=2))
            res = np.mean(res)
        res_l.append(res)
        
        # PSNR
        mse = mse_loss(torch.Tensor(pred).flatten(), torch.Tensor(gt).flatten())
        #psnr = 20 * np.log10(255) - 10 * np.log10(mse)
        psnr = 20 * np.log10(1) - 10 * np.log10(mse)
        psnr_l.append(psnr)

        ssim_score = ssim(gt, pred, multichannel=True)
        ssim_l.append(ssim_score)

        dot = np.sum(gt * pred, axis = 2) 
        pred_l2 = np.sqrt(np.sum(pred ** 2, axis = 2))
        gt_l2 = np.sqrt(np.sum(gt ** 2, axis = 2))
        #print("dot, pred_l2, gt_l2",dot.shape, pred_l2.shape, gt_l2.shape)
        if use_mask:
            #print(mask.shape)
            sam = np.sum(np.arccos(dot/(pred_l2 * gt_l2 + 1e-1)))/(np.sum(mask)/4)
        else:
            sam = np.mean(np.arccos(dot/(pred_l2 * gt_l2 + 1e-1)))
        sam_l.append(sam)
        
    print("---------------------------------")
    print("MSE:", sum(res_l) / num)
    print("PSNR:", sum(psnr_l) / num)
    print("SSIM:", sum(ssim_l) / num)
    print("SAM:", sum(sam_l) / num)
    print("---------------------------------")
    
def val_metrics2(file, num, use_mask=False):
    gt_str = "results_ground_truth"
    pred_str = "results_outputs"
    input_str = "results_cloudys"
    #print(file)

#     print("input shape", inp.shape)

    res_l = []
    psnr_l = []
    ssim_l = []
    sam_l = []

    for idx in range(num):
        file0 = file % (pred_str, idx)
        pred = np.load(file0)
        pred = np.squeeze(np.transpose(pred, (0,2,3,1)))
#         pred = np.sqrt(pred)
        pred = np.nan_to_num(pred)
#         print("pred shape", pred.shape)

        file1 = file % (gt_str, idx)
        gt = np.load(file1)
        gt = np.squeeze(np.transpose(gt, (0,2,3,1)))
#         gt = np.sqrt(gt)
        gt = np.nan_to_num(gt)
#         print("gt shape", gt.shape)

        file2 = file % (input_str, idx)
        inp = np.load(file2)
        inp = np.squeeze(np.transpose(inp, (0,2,3,1)))
#         inp = np.sqrt(inp)
        
        if use_mask:
            mask_single = np.sum(inp, axis=2)
            mask_single = (mask_single == 0)
            mask_single = np.expand_dims(mask_single, 2)
            mask = np.concatenate((mask_single, mask_single, mask_single), axis=2)
            if inp.shape[2] != pred.shape[2]:
                mask = mask[:,:,start:end]
            gt = gt * mask
            pred = pred * mask
            
        res = gt - pred
        res = res ** 2
#         print("res shape", res.shape)
        
        mse_loss = nn.MSELoss()
        #print(pred.shape, gt.shape)
        loss = mse_loss(torch.Tensor(pred).flatten(), torch.Tensor(gt).flatten())
#         print(idx, loss)
        
        #print("loss dif", loss, np.mean(res))
        if use_mask:
            #res = np.sum(res)/(np.sum(mask)/4)
            res = np.sum(res)/(np.sum(mask))
        else:
            #res = np.mean(np.sum(res,axis=2))
            res = np.mean(res)
        res_l.append(res)
        
        # PSNR
        mse = mse_loss(torch.Tensor(pred).flatten(), torch.Tensor(gt).flatten())
        psnr = 20 * np.log10(1) - 10 * np.log10(mse)
        psnr_l.append(psnr)

        ssim_score = ssim(gt, pred, multichannel=True)
        ssim_l.append(ssim_score)

        dot = np.sum(gt * pred, axis = 2) 
        pred_l2 = np.sqrt(np.sum(pred ** 2, axis = 2))
        gt_l2 = np.sqrt(np.sum(gt ** 2, axis = 2))
        #print("dot, pred_l2, gt_l2",dot.shape, pred_l2.shape, gt_l2.shape)
        if use_mask:
#             print(np.arccos(dot/(pred_l2 * gt_l2 + 1e-5)).shape)
#             print(np.sum(mask_single))
            sam = np.sum(mask_single * np.arccos(dot/(pred_l2 * gt_l2 + 1e-5)))/(np.sum(mask_single))
        else:
            sam = np.mean(np.arccos(dot/(pred_l2 * gt_l2 + 1e-5)))
        sam_l.append(sam)
        
    print("---------------------------------")
    print("MSE:", sum(res_l) / num)
    print("PSNR:", sum(psnr_l) / num)
    print("SSIM:", sum(ssim_l) / num)
    print("SAM:", sum(sam_l) / num)
    print("---------------------------------")
    
    #return res_l

# visualize inpainted images
def visualize_ls2(file, model_name, idx, stretch=(2, 98), show_res=True, show_trip=True, save=False, show_hist=True, xlim=0.9, vmax=100, threshold=0.05, save_res=False):
    gt_str = "results_ground_truth"
    pred_str = "results_outputs"
    input_str = "results_cloudys"
    sentinel_str = "results_sentinels"
    landsat_str = "results_landsats"

    file0 = file % (pred_str, idx)
    
    pred = np.load(file0)
    pred = np.squeeze(np.transpose(pred, (0,2,3,1)))

    pred = np.nan_to_num(pred)

    file1 = file % (gt_str, idx)
    gt = np.load(file1)
    gt = np.squeeze(np.transpose(gt, (0,2,3,1)))

    
    file2 = file % (input_str, idx)
    inp = np.load(file2)
    inp = np.squeeze(np.transpose(inp, (0,2,3,1)))

    
    file3 = file % (sentinel_str, idx)
    sen = np.load(file3)
    sen = np.squeeze(np.transpose(sen, (0,2,3,1)))

    file4 = file % (landsat_str, idx)
    land = np.load(file4)
    land = np.squeeze(np.transpose(land, (0,2,3,1)))

    
    
    mask = (inp == 0)
    if inp.shape[2] != pred.shape[2]:
        mask = mask[:,:,start:end]
        
    gt_res = gt * mask
    pred_res = pred * mask
    if show_res: 
        diff = np.abs(gt_res-pred_res)
    #     diff = diff[:,:,1:4]
        print("diff",diff.shape)
        plt.axis('off')
        diff = np.sum(diff, axis = 2)
        diff = diff/3
        ax = sns.heatmap(diff, vmin = 0, vmax = vmax, annot = False, xticklabels=False, yticklabels=False, cbar=False,square=True)
        if save_res:
            plt.savefig("res_" + model_name + str(idx) + ".jpg")
        plt.show()
        
        dot = np.sum(gt * pred, axis = 2)
        pred_l2 = np.sqrt(np.sum(pred ** 2, axis = 2))
        gt_l2 = np.sqrt(np.sum(gt ** 2, axis = 2))
        sim = dot/(pred_l2 * gt_l2 + 1e-5)
        print("dot, pred_l2, gt_l2",dot.shape, pred_l2.shape, gt_l2.shape)
        sim = sim * mask[:,:,0] + (1. - mask[:,:,0])
        m = copy.deepcopy(sim)
        
        print(sim.mean())
        m[sim>=threshold]=1
        m[sim<=threshold]=0

        #ax = sns.heatmap(m, vmin = 0, vmax = 1, annot = False,xticklabels=False, yticklabels=False, cmap = 'Greys')
        if save_res:
            plt.imsave("thres_" + model_name + str(idx) + ".jpg", m, dpi=150, cmap = 'Greys')
        plt.show()
        
    if show_trip:
        pred = pred * mask + gt * (1. - mask)
        pred = min_max_stretch(pred, stretch[0], stretch[1])
        gt = min_max_stretch(gt)
        inp = np.where(mask, 1, gt)
#         inp = min_max_stretch(inp, 0, 98)
        sen = min_max_stretch(sen)
        land = min_max_stretch(land)
        
        fig = plt.figure()
        gs = fig.add_gridspec(1, 5, hspace=0.1, wspace=0.1)
        axs = gs.subplots(sharex=False, sharey=False)
#         fig.suptitle('Sharing both axes')
        for i in range(5):
            axs[i].autoscale()
            axs[i].axis('off')

        axs[0].imshow(inp)
        axs[1].imshow(gt)
        axs[2].imshow(pred)
        axs[3].imshow(sen)
        axs[4].imshow(land)
#         axs[4].imshow(land.astype(np.uint8))
        plt.show()
    
        if save:
            plt.imsave("cloudy_" + str(idx) + ".jpg", inp, dpi=150)
            plt.imsave("ground_truth_" + str(idx) + ".jpg", gt, dpi=150)
            plt.imsave("landsat_" + str(idx) + ".jpg", land, dpi=150)
            plt.imsave("sentinel_" + str(idx) + ".jpg", sen, dpi=150)
            plt.imsave("pred_" + model_name + "_" + str(idx) + ".jpg", pred, dpi=150)
    
    if show_hist:
        # the metric for the histogram is mse, i. e., the l2-model of the difference vector
        diff = gt_res - pred_res
        mod = np.sqrt(np.sum(diff ** 2, axis = 2))
#         dot = np.sum(gt_res * pred_res, axis = 2)
#         pred_l2 = np.sqrt(np.sum(pred_res ** 2, axis = 2))
#         gt_l2 = np.sqrt(np.sum(gt_res ** 2, axis = 2))
        print("diff, mod",diff.shape, mod.shape)
        # sim = dot/(pred_l2 * gt_l2 + 1e-5)
#         proj = dot/(gt_l2 + 1e-5)
#         diff = np.absolute(proj - gt_l2)
        
        fig, ax = plt.subplots()
#         sns.histplot(sim.flatten(), bins = 10000, ax=ax)
        sns.histplot(mod.flatten(), bins = 10000, ax=ax)
        ax.grid(axis='y')
        ax.yaxis.grid(True) # Hide the horizontal gridlines
        ax.xaxis.grid(True)
        
        plt.xlim([0,xlim])
        plt.ylim([0,120])
        plt.xlabel("Pixel-wise vector difference (Cloudy Only)")
        plt.axvline(x=np.median(mod[np.logical_and(0 < mod, mod < xlim)]),
            color='red',ls='--')
        plt.text(np.median(mod[np.logical_and(0 < mod, mod < xlim)]),80,'%0.5f' % np.median(mod[np.logical_and(0 < mod, mod < xlim)]))
        plt.show()
        