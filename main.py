import cv2
import numpy as np
import time

videofile = "test.mp4"
templatefile = "template3.jpg"
findslen = 160

# template = cv2.imread('template3.jpg',cv2.IMREAD_GRAYSCALE)
template = cv2.imread(templatefile, 0)
dimfactor = 1
dim = (int(template.shape[0]/dimfactor), int(template.shape[1]/dimfactor))

template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
weights = cv2.VideoCapture(videofile)
# weights.read()
# weights = weights.set(3, weights.get(3)/2)
# weights = weights.set(4, weights.get(4)/2)

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
method = eval(methods[3])
cv2.imshow('image', template)
cv2.waitKey(0)

finds = np.empty(findslen, dtype=tuple)
findsloc = 0

while(weights.isOpened()):
    ret, colorframe = weights.read()
    colorframe = cv2.rotate(colorframe, cv2.ROTATE_90_CLOCKWISE)

    frame = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    midle = (top_left[0] + dim[0]//2, top_left[1] + dim[1]//2)
    finds[findsloc] = midle
    findsloc = (findsloc+1) % findslen

    for find in finds:
        cv2.circle(colorframe, find, 5, (25,25,255), -1, 10, 0)

    colorframe = cv2.resize(colorframe, (540, 960))
    cv2.imshow('frame', colorframe)
    # cv2.resizeWindow('frame', 1920/2, 1080/2)

    # if cv2.waitKey(1) & 0xFF == ord('p'):
    #     time.sleep(3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


weights.release()
cv2.destroyAllWindows()
