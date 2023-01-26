params = {}
params['color_space'] = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 9  # HOG orientations
params['pix_per_cell'] = 4  # HOG pixels per cell
params['cell_per_block'] = 2  # HOG cells per block
params['hog_channel'] = 'ALL'  # Can be 0, 1, 2, or "ALL"
params['spatial_size'] = (16, 16)  # Spatial binning dimensions
params['hist_bins'] = 16  # Number of histogram bins
params['spatial_feat'] = False  # Spatial features on or off
params['hist_feat'] = False  # Histogram features on or off
params['hog_feat'] = True  # HOG features on or off
params['size_of_window']=(108,36,3)
params['test_size']=0.8


win_size={}
s=[0.45, 0.611, 0.772, 0.933, 1.094, 1.255, 1.416, 1.577, 1.738, 1.899, 2.06]
for x,y in enumerate(s):
    win_size[f'scale_{x}']=(0,1000,y)
win_size['thresh']=8
win_size['overlap_thresh']= 0.05
# 0.5, 1.3
win_size['use_scale']=(2,3)