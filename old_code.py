# old code

# see https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
# ngl the tutorial sucks in terms of intuition about parameters: if you find
# something, please link it
def find_corners(img, blockSize, sobel_k=3, harris_k=0.04, thresh=0.35):
	# corners is a grayscale image where higher values are more probably corners.
	corners = cv.cornerHarris(img, blockSize, sobel_k, harris_k)

	#TODO: we could do something more robust by trying a range of threshold values and picking
	# the "best" one, i.e. the largest one that still gives all 4 corners.
	bin_corners = (corners > thresh * corners.max())

	# see the body of segment_adaptive for an explanation of what this does
	connectivity = 8
	num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(np.uint8(bin_corners), connectivity , cv.CV_32S)
	assert num_labels == 5

	# sorting connected components by area, getting rid of background component
	areas = sorted([(i, stat, (labels == i), centroids[i]) for i, stat in enumerate(stats)], key = lambda x:x[1][cv.CC_STAT_AREA])[:-1]
	#def proc_func(area):
		#return np.unravel_index(np.argmax(area[2], axis=None), area[2].shape)
		#return [area[3][1], area[3][0]]
		#return area[3]

	#corner_coords = [proc_func(area) for area in areas]

	# upper left corner should have the smallest sum of row and column
	ul_ind = max(range(len(areas)), key= lambda i: -sum(areas[i][3]))
	#upper right has a large column, small row
	ur_ind = max(range(len(areas)), key= lambda i: areas[i][3][0] - areas[i][3][1])
	#lower left has large row, small column
	ll_ind = max(range(len(areas)), key= lambda i: areas[i][3][1] - areas[i][3][0])
	#lower right has large row, large column
	lr_ind = max(range(len(areas)), key= lambda i: sum(areas[i][3]))

	all_inds = [ul_ind, ur_ind, lr_ind, ll_ind]

	assert len(set(all_inds)) == 4, str(all_inds)
	ul_coords = [areas[ul_ind][1][cv.CC_STAT_TOP], areas[ul_ind][1][cv.CC_STAT_LEFT]]
	ur_coords = [areas[ur_ind][1][cv.CC_STAT_TOP], areas[ur_ind][1][cv.CC_STAT_LEFT] + areas[ur_ind][1][cv.CC_STAT_WIDTH]]
	ll_coords = [areas[ll_ind][1][cv.CC_STAT_TOP] + areas[lr_ind][1][cv.CC_STAT_HEIGHT], areas[ll_ind][1][cv.CC_STAT_LEFT]]
	lr_coords = [areas[lr_ind][1][cv.CC_STAT_TOP] + areas[lr_ind][1][cv.CC_STAT_HEIGHT], areas[lr_ind][1][cv.CC_STAT_LEFT] + areas[lr_ind][1][cv.CC_STAT_WIDTH]]

	corner_coords = [ul_coords, ur_coords, lr_coords, ll_coords]
	print(corner_coords)
	print(img.shape)
	return corner_coords

"""
	neg = 1 - adap_mean
	#plt.imshow(neg, 'gray')
	#plt.show()
	ignore_calibration_item = gray * neg #+ (gray[inds[0]][inds[1]]) * adap_mean

	#print(stats(ignore_calibration_item))
	#print(ignore_calibration_item[:5,:5])
	table_mask = segment_table(ignore_calibration_item)
"""

"""
	#from main_test() in segment.py
	first_thresh = cv.adaptiveThreshold(dsk_cut_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 0)
	#imshow(first_thresh, 'gray')

	blockSize = 10
	kernel = np.ones((blockSize,blockSize),np.uint8)
	closed = cv.morphologyEx(np.float32(first_thresh), cv.MORPH_CLOSE, kernel)
	dsk_cut_mask = fill_holes(np.uint8(closed))
	#imshow(dsk_cut_mask, 'gray')

	connectivity = 8
	num_labels, labels, statistics, centroids = cv.connectedComponentsWithStats(dsk_cut_mask, connectivity, cv.CV_32S)

	# sorting connected components by area
	areas = sorted([(i, stat[cv.CC_STAT_AREA], centroids[i]) for i, stat in enumerate(statistics)], key = lambda x:x[1])

	piece = areas[-2] # the biggest area is the background, typically.
	#print(stats)
	#print(centroids[:5])
	dsk_cut_mask = (labels == piece[0])
	#kernel = np.ones((30,30))
	#dsk_cut_mask =  dsk_cut_mask * cv.morphologyEx(np.float32(dsk_cut_mask > 0), cv.MORPH_CLOSE, kernel)
	#plt.imshow(dsk_cut_mask)
	#plt.show()

	#print(stats(np.uint8(dsk_cut_mask)))
	ct = cv.findNonZero(dsk_cut_mask)
	x,y,w,h = cv.boundingRect(ct)

	final_cut = dsk_cut_img[y:y+h, x:x+w]

	imshow(final_cut, 'gray')
"""


"""
old segment_reference from segment.py

	else:

		img = imread('./raw_img_data/puzzle_pieces.png',0)
		img = cv.medianBlur(img,5)

		ret,th1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
		th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
		th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

		titles = ['Original Image', 'Global Thresholding (v = 127)',
	            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
		images = [img, th1, th2, th3]

		contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(img, contours, -1, (0,255,0), 3)

		for i in range(4):
		    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		    plt.title(titles[i])
		    plt.xticks([]),plt.yticks([])
		if cant_plot:
			plt.savefig('tmp.png')
		else:
			plt.show()

"""


"""
def argmax_convolve(rot_piece_color2, ref_img_color, possible_locs, i):

    #print('starting argmax convolve')

    # confidences should be a sequence of array-like things that can be broadcasted.
    # this function does some kind of element-wise aggregation to get a matrix of the
    # broadcast shape, or otherwise a scalar if a list of scalars was passed in.
    # examples are: sum, element-wise max, sum of squares, etc
    def confidence_aggregator(confidences):
        return np.amax(np.stack(confidences, axis=2), axis=2)
        #return sum(confidences)/3
        #return sum([conf * conf for conf in confidences])/10
        #return np.amin(np.stack(confidences, axis=2), axis=2)

    channels = ['r','g','b']
    all_confidences = []


    rot_piece_color = np.float32(rot_piece_color2)/(1 + np.sum(rot_piece_color2, axis=(0,1), keepdims=True))

    all_conv_res = []
    for dim_ind in range(ref_img_color.shape[2]):
        ref_img = ref_img_color[:,:,dim_ind]
        rot_piece = rot_piece_color[:,:,dim_ind]
        all_conv_res.append(cv.filter2D(ref_img, -1, rot_piece))
    #print('finished convolve')
    conv_res = confidence_aggregator(all_conv_res)
    assert conv_res.shape == ref_img.shape, str('shapes differ: conv is {}, ref is {}'.format(conv_res.shape, ref_img.shape))
    poss_locs_mask = np.zeros_like(conv_res)

    if possible_locs is None:
        inds = np.unravel_index(np.argmax(conv_res), conv_res.shape)
        confidence = conv_res[inds[0]][inds[1]]
    else:
        inds = (-1, -1)
        confidence = -1
        for loc in possible_locs:
            poss_locs_mask[loc[0]:loc[0] + loc[2], loc[1]:loc[1]+loc[2]] = 1

            for row_ind in range(loc[0], loc[0]+loc[2]):
                for col_ind in range(loc[1], loc[1] + loc[2]):
                    #print(row_ind, col_ind)
                    tmp_conf = conv_res[row_ind][col_ind]
                    if tmp_conf > confidence:
                        confidence = tmp_conf
                        inds = (row_ind, col_ind)

            #print(loc)
            #section = conv_res[loc[0]:loc[0] + loc[2], loc[1]:loc[1]+loc[2]]
            #box_inds = np.unravel_index(np.argmax(section), section.shape) 
            #global_inds = np.array(box_inds) + np.array([loc[0], loc[1]])
            #tmp_confidence = conv_res[global_inds[0]][global_inds[1]]
            #if tmp_confidence > confidence:
            #    confidence = tmp_confidence
            #    inds = global_inds
    conv_res_masked = conv_res * poss_locs_mask
    ref_img_masked = ref_img * poss_locs_mask

    if i == 45:
        images = [conv_res, ref_img_color,rot_piece_color2]
        titles = ['convolution: max conf {} {}'.format(confidence, i), 'reference', 'piece']
        for i in range(len(images)):
            imshow(images[i],title=titles[i], inds=None if i==2 else inds)


    #images = [conv_res, ref_img, rot_piece]
    #titles = ['convolution: max conf {}'.format(confidence), 'reference', 'piece']
    #for i in range(len(images)):
    #    plt.subplot(2,len(images)//2 + 1,i+1),plt.imshow(images[i],'gray')
    #    if i < 2:
    #        plt.subplot(2,len(images)//2 + 1,i+1),plt.plot(inds[1], inds[0], 'r+')
    #    plt.title(titles[i])
    #    #plt.xticks([]),plt.yticks([])

    #plt.show()

    return inds, confidence
    #return position, confidence

"""
