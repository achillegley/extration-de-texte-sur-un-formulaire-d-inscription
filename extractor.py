
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from pdf2image import convert_from_path , convert_from_bytes
import os
import pickle
import pytesseract
from PIL import Image, ImageDraw, ImageFont

def charge_base_position():
    with open("models/data_positions.pickle", "rb") as file:
        base_positions = pickle.load(file)
    return base_positions

#fonction de conversion de pdf en images
def convertPdfToImage(pdfPath, type):
  images = convert_from_bytes(open(pdfPath, 'rb').read(), size=2000)
  imagesNames={}
  for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save(str(type)+'/page'+ str(i) +'.png')
    imagesNames['page'+ str(i)]=Image.open(str(type)+'/page'+ str(i) +'.png')
  return imagesNames,images


#fonction de correpondance entre zones
def inRect(x1,y1,x2,y2,x,y):
 return (x1 < x < x2) and (y1 < y < y2)


# fonction de detection des zones de remplissage
def getzones(afterPath):
    before = cv2.imread("base_" + str(afterPath))
    after = cv2.imread(afterPath)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(before_gray, after_gray, full=True)

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    # i=0;
    textPositions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            textPositions.append(str(x) + '_' + str(y) + '_' + str(w) + '_' + str(h))
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return textPositions, after, score * 100

#fonction de calcul de distance entre 2 points
def custom_dist(x1,y1,x2,y2):
  return np.linalg.norm(np.array((x1,y1))-np.array((x2,y2)))

#fonction de calcul de distance entre 2 rectangles
def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return custom_dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return custom_dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return custom_dist(x1b, y1, x2, y2b)
    elif right and top:
        return custom_dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.

#fonction de calcul de la distance la plus proche
def get_closest_position(x1, y1, x1b, y1b,page_positions):
  min_distance=1000
  min_position_name=""
  for key in page_positions:
    key_split=key.split('_')
    current_distance=rect_distance(
        x1, y1, x1b, y1b,
        float(key_split[0]),
                      float(key_split[1]),
                      float(key_split[0])+float(key_split[2]),
                      float(key_split[1])+float(key_split[3]))
    if( current_distance<min_distance):
      min_distance=current_distance
      min_position_name=key
  return min_distance,min_position_name

#fonction de nommage des positions
def getPositionsNameBis(positions, base_positions):
  positionsNames={}
  score={}
  isInRect=False
  for i in positions:
    text_split=i.split('_')
    _,position_name=get_closest_position(float(text_split[0]),
                      float(text_split[1]),
                      float(text_split[0])+float(text_split[2]),
                      float(text_split[1])+float(text_split[3]),base_positions)
    if base_positions[position_name] in score:
          score[base_positions[position_name]]+=1
    else:
      score[base_positions[position_name]]=0
    positionsNames[i]=base_positions[position_name]+'_'+str(score[base_positions[position_name]])
  return positionsNames
#definition de la fonction
def getAllZones(imagesNames):
  positions={}
  for imagesName in imagesNames:
    positions[imagesName],_,_=getzones("pages/"+str(imagesName)+".png")
  return positions


#fonction de nommage des positions
def getPositionsName(positions, base_positions):
  positionsNames={}
  score={}
  isInRect=False
  for i in positions:
    for key in base_positions:
      key_split=key.split('_')
      #print(key_split)
      text_split=i.split('_')
      #print(text_split)
      isInRect=inRect(float(key_split[0]),
                      float(key_split[1]),
                      float(key_split[0])+float(key_split[2]),
                      float(key_split[1])+float(key_split[3]),
                      int(text_split[0]),
                      int(text_split[1]))
      if isInRect:
        if base_positions[key] in score:
          score[base_positions[key]]+=1
        else:
          score[base_positions[key]]=0
        positionsNames[i]=base_positions[key]+'_'+str(score[base_positions[key]])
        break;
  return positionsNames


#fonction d'obtention des labels de toutes les positions
def getAllPositionNames(positions,base_positions):
  positionsNames={}
  for key in positions:
    positionsNames[key]=getPositionsNameBis(positions[key], base_positions['positions/'+str(key)+'/'])
  return positionsNames


#fonction de crop d'images en fonction de la position
def cropImageByPosition(positions,currentImage,currentImagesFolder):
  currentCropped={}
  for i in positions:
    x=i.split('_')[0]
    y=i.split('_')[1]
    w=i.split('_')[2]
    h=i.split('_')[3]
    img_name=positions[i]
    im1 = currentImage.crop((float(x), float(y), float(x) + float(w), float(y) + float(h)))
    im1.save(currentImagesFolder+img_name +'.png')
    currentCropped[img_name]=Image.open(currentImagesFolder+img_name +'.png')
  return currentCropped


#coupure des pages
def cropAllImageByPosition(imagesNames,positionsNames):
  croppedImages={}
  for imageName in imagesNames:
    croppedImages[imageName]=cropImageByPosition(positionsNames[imageName],imagesNames[imageName],"cropped/"+imageName+"/")
  return croppedImages


#detection des images
def extract_text(croppedImages):
  results={}
  for croppedImage in croppedImages:
    currentResults={}
    for i in croppedImages[croppedImage]:
      splited_values=i.split('_')
      current_value=splited_values[0]
      for j in range(len(splited_values)-2):
        if(j>0):
          current_value+="_"+splited_values[j]
      detected=pytesseract.image_to_string(croppedImages[croppedImage][i],
                                           config='--psm 10 --oem 3 -c tessedit_char_whitelist=+-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvxyz0123456789')
      detected=detected.replace('\r', '').replace('\n', '').replace('\x0c',' ').replace(';','')
      if current_value in currentResults:
        currentResults[current_value]=detected + currentResults[current_value]
      else:
        currentResults[current_value]=detected
    results[croppedImage]=currentResults
  return results



def ordered_function(form_pdf):
    base_positions = charge_base_position()
    #conversion du formulaire en images
    imagesBaseNames, imagesBase = convertPdfToImage("uploads/application-form-vnu-ifi-2020-sim.pdf", "base_pages")
    imagesNames, images = convertPdfToImage(form_pdf, "pages")
    positions = getAllZones(imagesNames)
    positionsNames = getAllPositionNames(positions,base_positions)
    croppedImages = cropAllImageByPosition(imagesNames, positionsNames)
    results=extract_text(croppedImages)
    return results