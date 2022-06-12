#############################
######### IMPORTS ###########
#############################

from flask import Flask
from flask import render_template
from flask import request
from flask import redirect, url_for, render_template
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os



#############################
######### VARIABLES #########
#############################

app = Flask(__name__) #inicjalizacja aplikacji
UPLOAD_FOLDER = os.getcwd() + "/static/"#zmienna globalna. Dane aplikacji umieszczone w folderze static
MODELS_FOLDER = UPLOAD_FOLDER + "MODELS/"
STATIC_FOLDER = "static/"
MODEL = load_model(MODELS_FOLDER + "klasyfikacja128.h5")

#############################
######### FUNCTIONS #########
#############################

def predict(image_path, model):
    zmiana_plik = image_path
    zmiana = image.load_img(zmiana_plik, target_size = (128,128))
    zmiana = image.img_to_array(zmiana)
    zmiana = np.expand_dims(zmiana, axis = 0)
    zmiana = zmiana/255
    predict_x=model.predict(zmiana) 
    classes_x=np.argmax(predict_x,axis=1)
    probability = max(predict_x[0])
    def switch_demo(argument):
        switcher = {
            0: "Actinic keratoses - Zrogowacenie",
            1: "Basal cell carcinoma - rak podstawnokomórkowy ",
            2: "Benign keratosis-like - łagodna zmiana skórna",
            3: "Dermatofibroma - włókniak skórny",
            4: "Melanoma - czerniak",
            5: "Nevus - znamię melanocytowe",
            6: "Vascular lesion - zmiana naczyniowa",}
        return switcher.get(argument, "nothing")
    
    prediction = switch_demo(classes_x[0]) 
    return prediction, probability 

def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return render_template("klasyfikacja.html", user_image = full_filename)

###############
### ROUTING ###
###############

@app.route("/")
def main_page():
    if request.method == "POST":   
        image_file=request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            return render_template("index.html", prediction=pred, user_image = STATIC_FOLDER + image_file.filename)
        
    return render_template("index.html", prediction=0)#(plik główny, argument albo argumenty)

@app.route('/index', methods=['GET','POST'])
def main_page_redirect():
    return redirect(url_for('main_page'))

@app.route('/klasyfikacja', methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":   
        image_file=request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            print(MODEL)
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            prob = pred[1]
            pred = pred[0]
            return render_template("klasyfikacja.html", 
                                   prediction=pred,
                                   user_image = STATIC_FOLDER + image_file.filename,
                                   pewnosc = prob
                                  )
        
    return render_template("klasyfikacja.html", prediction=0)#(plik główny, argument albo argumenty)




#######################
######### MAIN ########
#######################

if __name__ == "__main__":
    
    app.run(port = 12000, debug = True)
    
