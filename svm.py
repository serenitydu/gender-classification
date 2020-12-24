import sys, os, cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

fPath = r'C:\Users\seren\OneDrive\Desktop\CSE5522\project\Train\female_front_rotate'
mPath = r'C:\Users\seren\OneDrive\Desktop\CSE5522\project\Train\male_front_rotate'


x = []
labels = []
bin_n = 256 # Number of bins
h = []
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

#hog = cv2.HOGDescriptor()

folder = os.fsencode(fPath)
for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith('.jpeg'):
             im = cv2.imread((fPath +'\\' + filename))
             h = hog(im)
             x.append(h)
             labels.append(1)

folder = os.fsencode(mPath)
for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith('.jpeg'):
             im = cv2.imread((mPath +'\\' + filename))
             h = hog(im)
             x.append(h)
             labels.append(-1)
print('Size: {}'.format(len(h)))
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.1, random_state=0)

print('Training Linear SVM')
# Create a linear SVM classifier 
#clf = svm.SVC(kernel='linear')
#clf = svm.SVC(C = 10.0, kernel='rbf', gamma=0.1)
clf = svm.SVC(kernel='poly',degree=3,gamma=1,coef0=0)

# Train classifier 
clf.fit(X_train, y_train)

# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))