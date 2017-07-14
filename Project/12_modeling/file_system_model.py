import tensorflow as tf
import os
import json

#to copy files from source dir to target dir
#ver 1.0
def copyFiles(sourceDir,  targetDir):
    if sourceDir.find(".csv") > 0:
        print 'error'
        return
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir,  file)
        targetFile = os.path.join(targetDir,  file)
        if os.path.isfile(sourceFile):
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
            if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                    open(targetFile, "wb").write(open(sourceFile, "rb").read())
        if os.path.isdir(sourceFile):
            First_Directory = False
            copyFiles(sourceFile, targetFile)

# ensure the path exist
#will reset the dir
#ver 1.0
def del_and_create_dir(dir_path):
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)
    tf.gfile.MakeDirs(dir_path)

#to save the dictionary to json file
#ver 1.0
def save_dic_to_json(dic, filename):
    with open(filename, 'w') as f:
        json.dump(dic, f)

#to read json file and save to a dic
#ver 1.0
def read_json_to_dic(filename):
    with open(filename, 'r') as f:
        contents = json.load(f)
    return contents
