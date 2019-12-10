import cv2 as cv
import imutils
from matplotlib import pyplot as plt

class Piece:

    #Initializes a piece with the deskewed image of the piece.
    def __init__(self, img_path, cut_img=None):
        self.img = cv.imread(img_path)
        self.find_initial_position(self.img)
        self.cut_img = cut_img # temporary placeholder; this should be calculated via piece segmentation

    #Finds and sets the initial pixel position of piece from the piece image and reference image
    def find_initial_position(self, img):
        #self.init_pos = (0, 0)
        pass

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        # find the pixel location of the puzzle piece in the reference image
        # assume we have an instance variable named self.cut_img that has a small image of just
        # the puzzle.
        piece = self.cut_img
        n = 12
        max_ang = 360 # imutils uses degrees
        rotation_angles = [max_ang * i / n for i in range(n)]

        for i, rot in enumerate(rotation_angles):
            rot_piece = imutils.rotate_bound(piece, rot)
            #plt.imshow(rot_piece)
            #plt.title('Piece {} of {}'.format(i+1, n))
            #plt.show()
            #position, confidence = argmax_convolve(rot_piece, ref_img)

        #self.final_pos = (x, y)
        #self.rot_delta = 0
        pass

    #Places the piece using Baxter/Sawyer
    def place(self, init_pos, final_pos, rot_delta):
        pass



# origin is the pixel coordinates (x, y) of the origin of the table frame (extracted by calibrate_ppm)
# pixel_loc is the pixel coordinates of the pixel we want to determine
# ppm is pixels per meter, found in calibrate_ppm
def pixel_to_table_frame(origin, pixel_loc, ppm):
    pixel_diff = np.array(pixel_loc) - np.array(origin)
    # note that this ^^ implicitly assumes that the "vertical" of the image is the x axis of
    # the table frame
    return pixel_diff/ppm