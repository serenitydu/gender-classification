import sys, math
from PIL import Image
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import time

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

if __name__ == "__main__":

  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
  mypath="/Users/Home/Downloads/wiki_crop/female/"
  savepath="/Users/Home/Downloads/wiki_crop/female_front/"
  finish_size_h=70
  finish_size_v=80
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:]=="jpg" ]
  for file in onlyfiles:
    print(file)
    img = cv2.imread(join(mypath, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(face)==1:
      x,y,w,h=face[0]
      print(len(face))
      roi_gray = gray[y:y+h, x:x+w]
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

      eyes = eye_cascade.detectMultiScale(roi_gray)
      if(len(eyes)==2):
          x1,y1,w1,h1=eyes[0]
          x2,y2,w2,h2=eyes[1]
          if x1<x2:
              x2,y2,w2,h2,x1,y1,w1,h1=x1,y1,w1,h1,x2,y2,w2,h2
          cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,255,0),2)
          cv2.rectangle(img,(x+x2,y+y2),(x+x2+w2,y+y2+h2),(0,255,0),2)
          CropFace(Image.fromarray(gray), eye_left=(x+x2+w2/2,y+y2+h2/2), eye_right=(x+x1+w1/2,y+y1+h1/2), offset_pct=(0.35,0.4), dest_sz=(finish_size_h,finish_size_v)).save(savepath+file)
      #cv2.imshow('img',img)
      #cv2.waitKey(0)
