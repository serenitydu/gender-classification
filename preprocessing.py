import os
import shutil

cwd = os.getcwd()
print(cwd)
for folderName, subfolders, filenames in os.walk('lfw'):
    print(folderName)
    for id, filename in enumerate(filenames):
        if id == 0:
            print(filename)
            shutil.move(folderName + '\\' + filename,'dataset\\' + filename)
            