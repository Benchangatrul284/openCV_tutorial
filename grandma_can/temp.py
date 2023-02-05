import cv2
import numpy as np
import random
# read img
def read_img():

    img = cv2.imread("maxresdefault.jpg")


    resize1 = cv2.resize(img,(300,300))
    cv2.imshow("girl",resize1)
    cv2.waitKey(0)


    resize2 = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    cv2.imshow("girl",resize2)
    cv2.waitKey(0)

def capture_video():

    #cap = cv2.VideoCapture("video-sample-mp4.mp4")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    #ret is a boolean value indicating whether fetching next frame is successfull
    # if ret:
    #     cv2.imshow("frame",frame)
    # cv2.waitKey(0)

    #get the video by a while loop
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(0,0),fx = 1.5,fy = 1.5)
        if ret:
            cv2.imshow("video",frame)
            press = cv2.waitKey(10)
            if press == ord('q'):
                break
        else:
            break

#note that in cv2 is first y then x
def create_image():
    img = np.full((500,500,3),fill_value = 0,dtype=np.uint8)
    for y in range(300):
        for x in range(200):
            for c in range(3):
                img[y][x][c] = random.randint(0,255)
    new_img = img[50:100,200:300]
    print(new_img.shape)
    cv2.imshow("new_image",new_img)
    cv2.waitKey(0)

def show_image(name: str,img: np.ndarray):
    cv2.imshow(name,img)
    cv2.waitKey(0)


def common_func():
    img = cv2.imread("maxresdefault.jpg")
    show_image("ori",img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,ksize=(9,9),sigmaX=10,sigmaY=10)
    canny = cv2.Canny(img,threshold1=100,threshold2=200)
    show_image("canny",canny)
    #膨脹
    dil = cv2.dilate(src = canny,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    show_image("dil",dil)
    #江線條變細
    ero = cv2.erode(src = dil,kernel=np.ones((3,3),dtype=np.uint8),iterations=1)
    show_image("ero",ero)

capture_video()

