from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('form.html')

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        
        data = CustomData(
                        cap_shape=(request.form.get("cap_shape")),
                        cap_surface=(request.form.get("cap_surface")),
                        cap_color=(request.form.get("cap_color")),
                        bruises=(request.form.get('bruises')),
                        odor=(request.form.get("odor")),
                        gill_attachment=(request.form.get("gill_attachment")),
                        gill_spacing=(request.form.get("gill_spacing")),
                        gill_size=(request.form.get("gill_size")),
                        gill_color=(request.form.get("gill_color")),
                        stalk_shape=(request.form.get("stalk_shape")),
                        stalk_root=(request.form.get("stalk_root")),
                        stalk_surface_above_ring=(request.form.get("stalk_surface_above_ring")),
                        stalk_surface_below_ring=(request.form.get("stalk_surface_below_ring")),
                        stalk_color_above_ring=(request.form.get("stalk_color_above_ring")),
                        stalk_color_below_ring=(request.form.get("stalk_color_below_ring")),
                        veil_type=(request.form.get("veil_type")),
                        veil_color=(request.form.get("veil_color")),
                        ring_number=(request.form.get("ring_number")),
                        ring_type=(request.form.get("ring_type")),
                        spore_print_color=(request.form.get("spore_print_color")),
                        population=(request.form.get("population")),
                        habitat=(request.form.get("habitat"))


                        
        )

            
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_new_data)

        results = ''

        if pred == 0:
            results='mushroom is poisonous'
        else:
            results = 'edible mushroom'

        return render_template('form.html',final_result=results)
  




if __name__ =='__main__':
        app.run(host='0.0.0.0',port=5000,debug=True)