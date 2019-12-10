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