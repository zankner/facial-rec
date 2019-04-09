import cv2
import dlib
import numpy as np
from PIL import Image
import multiprocessing as mp
import pathos.pools as pp
import os

'''
Need to loop through each image and preform the following:
    1) Find the face in the image
        Using HOG for face detection
    2) Apply affine transform to normalize the data
'''

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)
    ])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis =0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

class Process(object):

    def __init__(self,data_dir,save_dir,imgDim):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.imgDim = imgDim
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.landmark_indices = np.array([39, 42, 57])
    
    def process_images(self):
        meta_dir = [sub_dir for sub_dir in os.listdir(self.data_dir)
                    if os.path.isdir(os.path.join(self.data_dir, sub_dir))]
        for img_class in meta_dir:
            imgs = [img_path for img_path in os.listdir(os.path.join(self.data_dir,img_class))
                    if os.path.isfile(os.path.join(self.data_dir,img_class,img_path))]
            for img in imgs:
                img_path = os.path.join(self.data_dir,img_class,img)
                with Image.open(img_path) as i:
                	try:
	                    face, img_array = self.detect_face(i)
	                    landmarks=self.find_landmarks(img_array,face)
                            aligned_img = self.align(self.imgDim, img_array, face, landmarks)
                            self.save(aligned_img,os.path.join(img_class, img))
	                except Exception as e:
	                	print("Warning: {}".format(e))


    def detect_face(self,img):
        assert img is not None
        try:
            img = np.array(img)
            detected_faces = self.face_detector(img, 1)
            if (len(detected_faces) >= 0):
                return max(detected_faces, key=lambda rect: rect.width() * rect.height()), img
            else:
                return [],img
        except Exception as e:
            print("Warning: {}").format(e)


    def find_landmarks(self,img,bb):
        assert img is not None
        assert bb is not None
        #print(type(img)) 
        landmarks = self.predictor(img,bb)
        return [(p.x, p.y) for p in landmarks.parts()]

    def align(self, imgDim, img, bb,landmarks,scale=1.0):
        assert imgDim is not None
        assert img is not None
        assert landmarks is not None

        landmarks = np.float32(landmarks)

        transform = cv2.getAffineTransform(landmarks[self.landmark_indices],
                self.imgDim * MINMAX_TEMPLATE[self.landmark_indices] * scale + self.imgDim * (1-scale) / 2)
        aligned = cv2.warpAffine(img, transform, (self.imgDim, self.imgDim))
        return aligned

    def save(self,img, img_path):
        img_save_dir = os.path.join(self.save_dir,img_path.split('/')[0])
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        open(os.path.join(img_save_dir,img_path.split('/')[1]),'a').close()
        cv2.imwrite(os.path.join(img_save_dir,img_path.split('/')[1]),img)

    def main(self):
        print('main')
        print(mp.cpu_count())
        pool = pp.ProcessPool(mp.cpu_count())
        pool.map(self.main, ())


p = Process('../data/raw','../data/processed',40)
p.process_images()
p.main()


