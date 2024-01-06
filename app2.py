import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request


app=Flask(__name__)

model=load_model("IBMDR.h5")
@app.route('/')
def index():
    return render_template("index.html")
    return render_template('register.html')
@app.route('/afterreg',methods=['POST'])
def afterreg():
    x = [x for x in request.form.values()]
    print(x)
    data = {
        '_id': x[1],
        'name': x[0],
        'pow':x[2]
    }
    print(data)
    query = {'_id': {'$eq': data['_id']}}

    docs = my_database.get_quesry_result(query)
    print(docs)

    print(len(docs.all()))
    if (len(docs.all()) == 0):
        url = my_database.create_document(data)
        return render_template('register.html', pred="Registration Successful, please login using your details")
    else:
        return render_template('register.html', pred="You are already a member,please login using your details")


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(299,299))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['0','1','2','3','4']
        text="The Classified RETINA is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()