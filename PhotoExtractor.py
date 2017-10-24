import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.random as random
import math

def imshow(img):
    """
    Show an OpenCV image
    """
    plt.figure(figsize=(25,15))
    plt.axis('off')
    # Adapt colormap for OpenCV
    if len(img.shape) > 2:
        plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
    else:
        plt.imshow( img )
    plt.show()

def scaleImage(img, max_size):
    """
    Scale the image to be not bigger than max_size, keep the ratio
    """
    height, width, _ = img.shape

    # Image already enough small
    if ( height < max_size and width < max_size ):
        return img, 1

    # Choose bigger side
    if ( height > width ):
        ratio = height / max_size
    else:
        ratio = width / max_size

    img_scale = cv2.resize( img, ( round( width/ratio ), round( height/ratio )) )
    return img_scale, ratio

def getDistance(line):
    """
    Line: (x1, y1, x2, y2)
    """
    return math.sqrt( math.pow(line[0]-line[2], 2) + math.pow(line[1]-line[3], 2) )


def isClose(val, ref, threshold=0.1):
    """
    Compare val and ref and return True if below the threshold
    """
    if ( abs(val-ref) > threshold ):
        return False
    else:
        return True

class Line:

    def __init__(self, tup):
        self.rho = tup[0]
        self.theta = tup[1]

    def __str__(self):
        return ('(%d,%f)'
                % (self.rho, self.theta))

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        if ( key == 0 ):
            return self.rho
        elif ( key == 1 ):
            return self.theta
        else:
            raise IndexError

# Library for working on line
def getLineEq( p1, p2 ):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = b * p1[1] + a * p1[0]
    return a, b, c

def intersection_bis(o1, p1, o2, p2):
    a1, b1, c1 = getLineEq( o1, p1 )
    a2, b2, c2 = getLineEq( o2, p2 )

    det = a1*b2 - b1*a2

    if ( abs(det) < 0.01 ):
        return False, (-1, -1)
    else:
        return True, (  round((b2*c1 - b1*c2)/det),
                        round((a1*c2 - a2*c1)/det) )

def getLinePoint( l ):
    a = np.cos(l.theta)
    b = np.sin(l.theta)
    x0 = a*l.rho
    y0 = b*l.rho
    x2 = int(x0 + 1600*(-b))
    y2 = int(y0 + 1600*(a))
    x1 = int(x0 - 1600*(-b))
    y1 = int(y0 - 1600*(a))
    return [x1,y1], [x2,y2]

def getInterPoint(col, l1, l2 ):
    o1, p1 = getLinePoint( l1 )
    o2, p2 = getLinePoint( l2 )

    return intersection_bis(o1, p1, o2, p2)

# Tets
# Both Should return (True, (2,0))
# Not the case with intersection
assert( intersection_bis([1,2], [3,-2], [1, -2], [3,2] ) == (True, (2,0)) )
assert( intersection_bis([1, -2], [3,2] , [1,2], [3,-2]) == (True, (2,0)) )

POINT = 0
RECT = 1

def getMin( r, ind ):
    mini = 3333333
    for p in r:
        if ( p[ind] < mini ):
            mini = p[ind]
    return mini

def getMax( r, ind ):
    maxi = 0
    for p in r:
        if ( p[ind] > maxi ):
            maxi = p[ind]
    return maxi

def updateROI(p_rois, p_roi, rect, max_height, max_width ):
    for i in range(len(p_rois)):
        # TODO auto threshold
        if ( getDistance( [p_rois[i][POINT][0], p_rois[i][POINT][1], p_roi[0], p_roi[1]] ) < 100 ):
                n_x_min = getMin(rect, 0)
                n_x_max = getMax(rect, 0)

                n_y_min = getMin(rect, 1)
                n_y_max = getMax(rect, 1)

                r = p_rois[i][RECT]
                x_min = getMin(r, 0)
                x_max = getMax(r, 0)

                y_min = getMin(r, 1)
                y_max = getMax(r, 1)

                x_min = min(x_min, n_x_min)
                x_max = max(x_max, n_x_max)

                y_min = min(y_min, n_y_min)
                y_max = max(y_max, n_y_max)

                if ( x_min < 0 ): x_min = 0
                if ( y_min < 0 ): y_min = 0
                if ( x_max > max_width ): x_max = max_width
                if ( y_max > max_height ): y_max = max_height

                p_rois[i][RECT] = [ (x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max) ]
                return

    # NEW
    p_rois.append( [p_roi, rect] )

def keepSmallerRect( rects, min_w, min_h ):
    if ( len(rects) == 0 ):
        return []
    smaller_r = rects[0]
    for rect in rects:
        n_x_min = getMin(rect, 0)
        n_x_max = getMax(rect, 0)
        n_y_min = getMin(rect, 1)
        n_y_max = getMax(rect, 1)

        x_min = getMin(smaller_r, 0)
        x_max = getMax(smaller_r, 0)
        y_min = getMin(smaller_r, 1)
        y_max = getMax(smaller_r, 1)

        x_min = max(x_min, n_x_min)
        x_max = min(x_max, n_x_max)
        y_min = max(y_min, n_y_min)
        y_max = min(y_max, n_y_max)

        smaller_r = [ (x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max) ]

    # Check it is not too small
    x_min = getMin(smaller_r, 0)
    x_max = getMax(smaller_r, 0)
    y_min = getMin(smaller_r, 1)
    y_max = getMax(smaller_r, 1)

    if ( x_max-x_min < min_w or y_max-y_min < min_h ):
        return []

    return smaller_r



def getROIfromRect( img, r ):
    x_min = getMin(r, 0)
    x_max = getMax(r, 0)

    y_min = getMin(r, 1)
    y_max = getMax(r, 1)

    return img[y_min:y_max, x_min:x_max, :]


class PhotoExtractor:

    def __init__(self, show=False):
        self.show = show

    def select_image(self, image_url, scan_format="a3", photo_format=""):
        # Clean
        self.segments = None
        self.img_segments = None
        self.hough_lines = None
        self.parallels = None
        self.perpendiculars = None
        self.img_hough = None
        self.rectangles = None
        self.img_rect = None
        self.img_pt_rois = None
        self.img_rois = None

        self.photos_segments = []
        self.photos_img_segments = []
        self.photos_hough_lines = []
        self.photos_img_hough = []
        self.photos_parallels = []
        self.photos_perpendiculars = []
        self.photos_rectangles = []
        self.photos_img_rect = []

        # Read
        self.img = cv2.imread(image_url)
        bordersize = 50
        # Add border for photos touching the side
        self.img = cv2.copyMakeBorder(self.img, top=bordersize, bottom=bordersize,
                          left=bordersize, right=bordersize,
                          borderType= cv2.BORDER_CONSTANT,
                          value=[255,255,255])
        # Scale
        self.img_scale, self.ratio = scaleImage(self.img, 1600)

        # Auto variables
        # TODO
        self.photo_size_w = round(2300/self.ratio)
        self.photo_size_h = round(1590/self.ratio)
        self.threshold_line_size = 150
        self.threshold_line_size_photo = 50
        self.threshold_hough_votes = 300
        self.threshold_hough_votes_photo = 100
        self.rho_step = 1
        self.theta_step = np.pi/180
        self.photo_margin_size = 10
        self.photo_margin_size_photo = 10
        self.threshold_grades = 0.01 # Around 1 grade


    def getSegmentsInImage(self, img_scale, threshold_line_size=40, kernel_dilate=np.ones((7, 7)),
            show=False):
        img_gray = cv2.cvtColor(img_scale,cv2.COLOR_BGR2GRAY)

        # Extract lines
        ls = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        lines = ls.detect(img_gray)
        lines = lines[0]

        if ( lines is None ):
            return [], np.zeros((img_scale.shape[0], img_scale.shape[1],3), np.uint8)

        lines = lines.reshape(-1, 4)

        # Sort lines
        lines_cut = []
        for line in lines:
            if ( getDistance(line) > threshold_line_size ):
                lines_cut.append(line)

        lines = np.array(lines_cut)

        # Create Image with segments
        img_segments = np.zeros((img_scale.shape[0], img_scale.shape[1],3), np.uint8)
        img_segments = ls.drawSegments( img_segments, lines)

        img_segments = cv2.dilate(img_segments, kernel_dilate)

        if ( show ):
            imshow(img_segments)

        return lines, img_segments

    def printHoughLines(self, img_hough_lines, hough_lines, col=(0,0,255), show=False):
        length_lines = max(img_hough_lines.shape)
        # Print Hough lines
        for rho,theta in hough_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + length_lines*(-b))
            y1 = int(y0 + length_lines*(a))
            x2 = int(x0 - length_lines*(-b))
            y2 = int(y0 - length_lines*(a))

            cv2.line(img_hough_lines,(x1,y1),(x2,y2),col,2)

        if ( show ):
            imshow(img_hough_lines)

    def getHoughLines(self, img_segments, img_ref, threshold_hough_votes=700, rho_step=1,
            theta_step=np.pi/180, show=False):
        """
        """
        if ( len(img_segments.shape) > 2 ):
            img_segments = cv2.cvtColor(img_segments,cv2.COLOR_BGR2GRAY)

        # Seek lines
        hough_lines = cv2.HoughLines(img_segments, rho_step, theta_step, threshold_hough_votes)

        if ( hough_lines is None ):
            return np.array([])

        hough_lines = hough_lines.reshape(-1, 2)


        if ( show ):
            img_hough = img_ref.copy()
            self.printHoughLines(img_hough, hough_lines, show=show)
            print(hough_lines.size)

        return hough_lines

    # Important for avoiding inside photo
    # Clean hough lines
    def cleanHoughLines(self, hough_lines, img_ref, photo_size_w, photo_size_h,
        photo_margin_size=10, threshold_grades=0.02, show=False, dont_clean = False):
        # Find parallels
        parallels = []

        for i in range(len(hough_lines)):
            l1 = Line(hough_lines[i])
            for j in range(i, len(hough_lines)):
                l2 = Line(hough_lines[j])
                # Parallel && separated with a certain size
                if (dont_clean
                    or (
                        isClose(l1.theta, l2.theta, threshold=threshold_grades)
                        and  (
                                isClose( abs(l1.rho-l2.rho), photo_size_w , threshold=photo_margin_size)
                                or isClose( abs(l1.rho-l2.rho), photo_size_h , threshold=photo_margin_size) )
                       )
                   ):
                    parallels.append([l1,l2])

        parallels = np.array(parallels)


        # Find perpendiculars
        perpendiculars = []

        for i in range(len(parallels)):
            p1 = parallels[i]
            for j in range(i, len(parallels)):
                p2 = parallels[j]
                if ( (isClose( abs( p1[0].theta - p2[0].theta), math.pi/2, threshold=threshold_grades))
                # Not square, only select rectangles
                and (not isClose( abs( p1[0].rho - p1[1].rho), abs( p2[0].rho - p2[1].rho), threshold=100)) ) :
                    perpendiculars.append([p1,p2])


        perpendiculars = np.array(perpendiculars)

        img_hough_clean = img_ref.copy()
        for per in perpendiculars:
            col = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))
            self.printHoughLines(img_hough_clean, per.flatten(), col=col, show=False)

        if ( show ):
            print(parallels.size)
            print('--------')
            print(perpendiculars.size)
            imshow(img_hough_clean)

        return parallels, perpendiculars, img_hough_clean

    def createRectangles(self, perpendiculars, img_ref, show=False):
        img_rect = img_ref.copy()
        rectangles = []

        # Print intersection
        for per in perpendiculars:
            col = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))
            l1 = per[0][0]
            l2 = per[0][1]
            l3 = per[1][0]
            l4 = per[1][1]
            pt = [None] * 4
            b = np.zeros(4)
            b[0], pt[0] = getInterPoint( col, l1, l3 )
            b[1], pt[1] = getInterPoint( col, l1, l4 )
            b[2], pt[2] = getInterPoint( col, l2, l3 )
            b[3], pt[3] = getInterPoint( col, l2, l4 )

            rectangles.append(pt)
            for i in range(b.size):
                if ( b[i] ):
                    cv2.circle(img_rect, pt[i], 20, col, thickness=-1)
            cv2.rectangle(img_rect, pt[0], pt[3], col, thickness=5)

        if ( show ):
            imshow(img_rect)
        return rectangles, img_rect

    def selectROIs(self, img_ref, rectangles, show=False):
        # Get average center for ROI and keep bigger area around
        img_pt_rois = img_ref.copy()

        # [pt, rect]
        rois = []

        for r in rectangles:
            p_roi1 = 0
            p_roi2 = 0
            for i in range(4):
                p_roi1 = p_roi1+ r[i][0]
                p_roi2 = p_roi2+ r[i][1]

            p_roi1 = round(p_roi1/4)
            p_roi2 = round(p_roi2/4)
            p_roi = (p_roi1, p_roi2)

            updateROI(rois, p_roi, r, img_ref.shape[0], img_ref.shape[1] )

        img_rois = []

        for roi in rois:
            img_roi = getROIfromRect( img_ref, roi[1] )
            bordersize=10
            img_roi = cv2.copyMakeBorder(img_roi, top=bordersize, bottom=bordersize,
                                  left=bordersize, right=bordersize,
                                  borderType= cv2.BORDER_CONSTANT,
                                  value=[255,255,255])
            img_rois.append( img_roi )

            col = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))
            cv2.circle(img_pt_rois, roi[0], 20, col, thickness=-1)
            cv2.rectangle(img_pt_rois, roi[1][0], roi[1][3], col, thickness=5)

        if ( show ):
            imshow(img_pt_rois)
        img_rois = np.array(img_rois)
        return img_pt_rois, img_rois


    def extract_photos(self, write=True, show=False):
        assert ( self.img_rois.size > 0 ), "No subimages to extract, do prepare_scan() before or check why no subimages are found"

        ind_photo = 1
        self.photos_segments = []
        self.photos_img_segments = []
        self.photos_hough_lines = []
        self.photos_img_hough = []
        self.photos_parallels = []
        self.photos_perpendiculars = []
        self.photos_rectangles = []
        self.photos_img_rect = []

        for img_roi in self.img_rois:

            # Line
            segments, img_segments =  self.getSegmentsInImage(img_roi,
                    threshold_line_size=self.threshold_line_size_photo,
                    kernel_dilate=np.ones((4, 4)), show=self.show)

            # Hough

            hough_lines = self.getHoughLines(img_segments,
                    img_roi, threshold_hough_votes=self.threshold_hough_votes_photo, rho_step=self.rho_step, theta_step=self.theta_step, show=self.show)

            # Rect
            parallels, perpendiculars, img_hough = self.cleanHoughLines(hough_lines, img_roi,
                    self.photo_size_w, self.photo_size_h, photo_margin_size=self.photo_margin_size_photo, threshold_grades=self.threshold_grades, show=self.show,
                    dont_clean = False)

            rectangles, img_rect = self.createRectangles(perpendiculars, img_roi,
                    show=self.show)

            # Selection of smaller one
            best_rect = keepSmallerRect( rectangles,
                round(self.photo_size_w/2), round(self.photo_size_h/2) )

            img_rect = img_roi.copy()
            if len(best_rect) > 3:
                col = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))
                for pt in best_rect:
                    cv2.circle(img_rect, pt, 20, col, thickness=-1)
                cv2.rectangle(img_rect, best_rect[0], best_rect[3], col,
                    thickness=5)

                # Extract
                x_min = getMin(best_rect, 0)
                x_max = getMax(best_rect, 0)

                y_min = getMin(best_rect, 1)
                y_max = getMax(best_rect, 1)

                if ( write ):
                    cv2.imwrite("photo_"+ str(ind_photo)+".jpg",
                        img_roi[y_min:y_max, x_min:x_max, :])
                    ind_photo = ind_photo + 1

                if ( show ):
                    print(best_rect)
                    imshow(img_rect)

            # Record steps
            self.photos_segments.append(segments)
            self.photos_img_segments.append(img_segments)
            self.photos_hough_lines.append(hough_lines)
            self.photos_img_hough.append(img_hough)
            self.photos_parallels.append(parallels)
            self.photos_perpendiculars.append(perpendiculars)
            self.photos_rectangles.append(rectangles)
            self.photos_img_rect.append(img_rect)


    def prepare_scan(self):
        self.segments, self.img_segments = self.getSegmentsInImage(self.img_scale,
            threshold_line_size=self.threshold_line_size, kernel_dilate=np.ones((7, 7)),
            show=self.show)

        self.hough_lines = self.getHoughLines(self.img_segments,
            self.img_scale,threshold_hough_votes=self.threshold_hough_votes, rho_step=self.rho_step,
            theta_step=self.theta_step, show=self.show)

        self.parallels, self.perpendiculars, self.img_hough = self.cleanHoughLines(self.hough_lines,
            self.img_scale, self.photo_size_w, self.photo_size_h, photo_margin_size=self.photo_margin_size,
            threshold_grades=self.threshold_grades, show=self.show, dont_clean=False)

        # Create and print rectangle
        self.rectangles, self.img_rect = self.createRectangles(self.perpendiculars,
            self.img_scale, show=self.show)

        self.img_pt_rois, self.img_rois= self.selectROIs(self.img_scale, self.rectangles,
            show=self.show)
