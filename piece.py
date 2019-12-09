import cv2 as cv

class Piece:

    #Initializes a piece with the deskewed image of the piece.
    def __init__(self, img_path):
        self.img = cv.imread(img_path)
        self.find_initial_position(self.img)

    #Finds and sets the initial pixel position of piece from the piece image and reference image
    def find_initial_position(self, img):
        #self.init_pos = (0, 0)
        pass

    #Solves and sets the final pixel position and rotation delta
    def solve_piece(self, ref_img):
        #self.final_pos = (x, y)
        #self.rot_delta = 0
        pass

    #Places the piece using Baxter/Sawyer
    def place(self, init_pos, final_pos, rot_delta):
        pass
