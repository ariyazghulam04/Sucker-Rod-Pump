import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd
import cv2
import streamlit as st

from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tempfile import NamedTemporaryFile

from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model


st.title('Dynagraph Interpreter and Pump Recommendation')

st.markdown('''

* This program is able to diagnose the SRP problem by reading the pump dynagraph 
* This problem is also able to give optimization recommendation to the SRP
* The program classify the dynagraph using deep machine learning algorithm (Tensorflow and Keras)


''')

st.write('**Dynagraph Interpreter**')

class_names = ['AirLock',
 'Broken Rod',
 'Gas Existence',
 'Inlet Valve Delay Closing',
 'Inlet Valve Leakage',
 'Outlet Sudden Unloading',
 'Outlet Valve Leakage',
 'Parrafin Wax',
 'Piston Sticking',
 'Plunger Fixed Valve Collision',
 'Plunger Guide Ring Collision',
 'Plunger Not Filled Fully',
 'Proper Work',
 'Rod Excesive Vibration',
 'Sand Problem',
 'Small Plunger',
 'Thick Oil',
 'Tubing Leakage',
 'Valve Leakage']

def import_and_predict(image_data, model):
    
    size = (180,180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
    img_reshape = img[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
        
    return prediction

#create layer for  deep learning model structure
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(19)
])

#define model cost function and metrics
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights('weights.hdf5')

st.sidebar.subheader('Dynagraph Image Input')

input = st.sidebar.selectbox('Choose Image: ',['Your Image','Sample Image'])

if input =='Your Image':
    uploaded_file = st.sidebar.file_uploader("Choose dynagraph file: ",type=['png','jpg'])

    if uploaded_file is None:
        image = Image.open('test/' + os.listdir('test')[0])
        st.image(image)
        
    else:
        st.image(uploaded_file)
        image = Image.open(uploaded_file)

else: 
    image_opt = st.sidebar.selectbox('Choose sample image: ',os.listdir('test'))
    
    image = Image.open('test/' + image_opt)
    st.image(image)


predictions = import_and_predict(image, model)
score = tf.nn.softmax(predictions[0])

st.success(
            "Based on the dynagraph, the pump most likely to have **{}** problem with a **{:.2f}** % confidence based on dataset."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))





st.sidebar.subheader('Pump Recommendation Input')

#Additional Parameter Input
OilGravity=st.sidebar.number_input("Oil Gravity (API) : ",value=30)
GasGravity=st.sidebar.number_input("Gas Gravity (API) : ",value=30)
PumpDepth=st.sidebar.number_input("Pump Depth (ft): ",value = 1999)
dpl=st.sidebar.number_input("Plunger Diameter (in.): ",value = 1)
dr=st.sidebar.number_input("Rod Diameter (in.): ",value=1)
RhoSteel=st.sidebar.number_input("Steel Density (lb/ft3): ", value = 122)
c=st.sidebar.number_input("C value (in.): ", value = 130)
h=st.sidebar.number_input("h value (in.): ", value = 10)
d1=st.sidebar.number_input("d1 (in.): ", value = 50)
d2=st.sidebar.number_input("d2 (in.): ", value = 10)
E=st.sidebar.number_input("Steel Modulus Young (psi): ", value = 20)
SurfaceStrokeLength=st.sidebar.number_input("Surface Stroke Length (in.): ", value = 30)
StrokeLength=st.sidebar.number_input("Theoritical Pump Stroke Length (in.): ", value = 40)
MaxStrokeLength=st.sidebar.number_input("Theoritical Pump maksimum Stroke Length (in.): ", value = 50)
DesignRate=st.sidebar.number_input("Design Rate (STB/ d): ", value = 60)
PumpSpeed=st.sidebar.number_input("Pump Speed (Strokes/ min): ", value = 100)

#SRP Unit Type
#print("Please choose your SRP Unit Type:")
#print("a. Conventional")
#print("b. Air-balanced")
Userinput=st.sidebar.selectbox("Please choose your SRP Unit Type: ",["Conventional","Air-balanced"])

if Userinput == 'Conventional':
    Userinput1 = 'a'
else:
    Userinput1 = 'b'

#PreCalculation
#Hydrocarbon Properties
SGoil=141.5/(OilGravity+131.5)
SGgas=141.5/(GasGravity+131.5)
Bo=1.3

#Area
Ar=(22/7)*dr*dr/4
Ap=(22/7)*dpl*dpl/4

#Pump Displacement
N= 0.1484*Ap*SurfaceStrokeLength*PumpSpeed
Kr=727.0008
PumpD= DesignRate*Bo//0.1484/Ap/SurfaceStrokeLength
#Fluid Load (Fo)
Fo=((0.8*SGoil*62.4)+(0.2*SGgas*0.08))*PumpDepth*Ap/144

#Weight Of Rods In Fluid (Wrf)
Wrf=RhoSteel*PumpDepth*Ar/144

#Total Load (Wrf + Fo)
TotalLoad=Wrf+Fo

#Volumetric Efficiency
Ev=DesignRate/N

#Actual Liquid Production Rate
ActualRate=0.1484*Ap*PumpD*StrokeLength*Ev/Bo

#Cyclic Load Factor
M=1+(c/h)

#Peak Polished Rod Load
#Conditional for Conventional and Air-balance
if Userinput1 == "a":
    F1=(StrokeLength*PumpD*PumpD*(1+(c/h)))/70471.2
elif Userinput1 == "b":
    F1=(StrokeLength*PumpD*PumpD*(1-(c/h)))/70471.2

PRLmax=Fo+((1.1+F1)*Wrf)

#Minimum Polished Rod Load
#Conditional for Conventional and Air-balance
if Userinput1 == "a":
    F2=(StrokeLength*PumpD*PumpD*(1-(c/h)))/70471.2
elif Userinput1 == "b":
    F2=(StrokeLength*PumpD*PumpD*(1+(c/h)))/70471.2

PRLmin=(0.65-F2)*Wrf

#Frictional Power
Pf=0.000000631*Wrf*SurfaceStrokeLength*(PumpD)

#Polished Rod Power
PRHP=SurfaceStrokeLength*PumpD*(PRLmax-PRLmin)/750000

#Name Plate Power
PNamePlate=Pf+PRHP

#Work Done by Pump
PumpWork=TotalLoad*Ev

#Work Done by Polished Rod
PRwork=((PRLmax+PRLmin)/2)+Fo+Wrf

#Pump Stroke Length
PumpStrokeLength=c*d2/d1

#Static Stretch
StaticStretch=Fo*PumpDepth/Ar/E

#Plunger Over Travel (EP)
EP=Fo*PumpDepth/Ap/E

#Maximum Torque
Torque=StrokeLength/4*((Wrf)+(2*StrokeLength*PumpD*PumpD*Wrf/70471.2))

#1/Kr
konstanta=1/Kr

#Fo/Skr
SKr=SurfaceStrokeLength/(konstanta)
X=Fo/(SKr)

#Counter Weight Required (CBE)
if Userinput1 == "a":
    CBE= (0.5*Fo)+Wrf*(0.9+(StrokeLength*PumpD*PumpD*c/(70471.2*h)))
elif Userinput1 == "b":
    CBE= (0.5*Fo)+Wrf*(0.9-(StrokeLength*PumpD*PumpD*c/(70471.2*h)))

#Counter Weight Position
CounterPosition=c

#Damping Factor
DampingFactor=(0.5+0.15)/2

#Stress (Max)
StressMax=PRLmax/Ar

#Stress (Min)
StressMin=PRLmin/Ar

#Recommendation Output
st.info('''

**Pump Recommendation**

Frictional Power               :  {} Horse Power

Polished Rod Power             :  {} Horse Power

Name Plate Power               :  {} Horse Power

Work Done By Pump              :  {} lbf

Work Done By Polished Rod      :  {} lbf

Volumetric Efficiency          :  {}

Actual Liquid Production Rate  :  {} STB/day

Cyclic Load Factor             :  {}

Peak Polished Rod Load         :  {} lbf

Minimum Polished Rod Load      :  {} lbf

Pump Stroke Length             :  {} in

Static Stretch                 :  {} in

Plunger OverTravel (EP)        :  {} in

Fluid Load (Fo)                :  {} lbf

Weight Of Rods in Fluid (Wrf)  :  {} lbf

Total Load (Wrf + Fo)          :  {} lbf

Maximum Torque                 :  {} lbf.ins

Fo/SKr                         :  {}

1/Kr                           :  {}

CounterWeight Required (CBE)   :  {} lbf

Counter Weight Position        :  {} in

Damping Factor                 :  {}

Maximum Stress                 :  {} psi

Minimum Stress                 :  {} psi


'''.format(Pf,PRHP,PNamePlate,PumpWork,PRwork,Ev,ActualRate,M,PRLmax,PRLmin,PumpStrokeLength,StaticStretch,EP,Fo,Wrf,
            TotalLoad,Torque,X,konstanta,CBE,c, DampingFactor, StressMax, StressMin))

####SECOND PROGRAM####
st.title('Pump Problem Detection')

df = pd.read_excel ('Input.xlsx', engine= 'openpyxl')

st.subheader('**Program Input**')

input = st.selectbox('Choose Input File: ',['Your File','Sample File'])

if input =='Your File':
    uploaded_file = st.file_uploader("Choose input file: ",type=['xlsx','xls'])

    if uploaded_file is None:
        df = pd.read_excel ('Input.xlsx', engine= 'openpyxl')
        
    else:
        df = pd.read_excel (uploaded_file, engine= 'openpyxl')
        

else: 
    file_opt = st.selectbox('Choose sample file input: ',['Input.xlsx'])
    
    df = pd.read_excel (file_opt, engine= 'openpyxl')




structure_opt = df['Structure'].unique().tolist()
structure_opt = [x for x in structure_opt if pd.isnull(x) == False]
Structure = st.selectbox("Choose your structure name: ",structure_opt)
#Structure = st.sidebar.text_input ("Type your structure name:")

well_opt = df[df['Structure'] == Structure]['Well Name']
well_opt = [x for x in well_opt if pd.isnull(x) == False]
Well = st.selectbox("Choose your well name:",well_opt)

location = df[(df['Structure']==Structure) & (df['Well Name']==Well)].index.values
location = location[0]

#Define Variable
#PSE Variable
RGL = df.at[int(location), 'Gas-Liquid Ratio\n(GLR)']
FVF = df.at[int(location), 'Formation Volume Factor\n(FVF)']
WF =  df.at[int(location), 'Water Cut\n(WF)']
Dp = df.at[int(location), 'Pump Diameter\n(Dp)']
SGl = df.at[int(location), 'Liquid SG\n(SGl)']
L = df.at[int(location), 'Sucker Rod Length\n(L)']
S = df.at[int(location), 'Stroke Length\n(S)']
ai = df.at[int(location), 'Ratio Sucker Rod and total length\n(ai)']
fri = df.at[int(location), 'Sucker Rod Area\n(fri)']
ft = df.at[int(location), 'Tubing Area\n(ft)']

#SP Variable
K1 = df.at[int(location), 'Weighting Coeff 1\n(K1)']
K2 = df.at[int(location), 'Weighting Coeff 2\n(K2)']
K3 =  df.at[int(location), 'Weighting Coeff 3\n(K3)']
K4 = df.at[int(location), 'Weighting Coeff 4\n(K4)']
LL = df.at[int(location), 'Liquid level\n(L)']
Q = df.at[int(location), 'Production Rate\n(Q)']
Lo = df.at[int(location), 'Gas Column @bottom dead center\n(Lo)']
Log = df.at[int(location), 'Gas Column @Top dead center\n(Log)']
PD = df.at[int(location), 'Plunger Displacement\n(PD)']
Er = df.at[int(location), 'Elasticity Modulus\n(Er)']
Ar = df.at[int(location), 'Rod Area\n(Ar)']
rhor = df.at[int(location), 'Rod Density\n(rhor)']
Lr = df.at[int(location), 'Rod Length\n(Lr)']
Fr = df.at[int(location), 'Rod Load\n(Fr)']
Med = df.at[int(location), 'Motor Driving Torque\n(Med)']
Mcsd = df.at[int(location), 'Crank Torque Std. Deviation\n(Mcsd)']
Angle = df.at[int(location), 'Motor Angle\n(Angle)']
Po = df.at[int(location), 'Motor Power without Load\n(Po)']
nh = df.at[int(location), 'Motor Rated Efficiency\n(nh)']
Ph = df.at[int(location), 'Motor Power with Load\n(Ph)']

#PUS Variable
load = df.at[int(location), 'Pumping Load\n(load)']
min_load = df.at[int(location), 'Min. Pumping Load\n(min_load)']
max_load =  df.at[int(location), 'Max. Pumping Load\n(max_load)']
PI = df.at[int(location), 'Productivity Index\n(PI)']
Pres = df.at[int(location), 'Reservoir Pressure\n(Pres)']
Pwf = df.at[int(location), 'Well Flowing Pressure\n(pwf)']

#Pumping System Efficiency
def PSE (RGL, FVF, WF, Dp, SGl, L, S, ai, ft, fri):
    temp = 100*((1.1/(1+RGL))-0.1)*(1/(FVF*(1-WF)+WF))*0.8924*(1-((Dp*Dp)*SGl*L/(2.62*(10**11)*S))*((ai/fri)+(1/ft)))
    return temp

PSEResult = PSE (RGL, FVF, WF, Dp, SGl, L, S, ai, ft, fri)
if (PSE (RGL, FVF, WF, Dp, SGl, L, S, ai, ft, fri)<40):
    ans = "Pumping System Not Efficient"
else:
    ans = "Pumping System Efficient Enough"

#Swabbing Parameter
def SP (K1, K2, K3, K4, LL, Q, Lo, Log, PD, Er, Ar, rhor, Lr, Fr, Mcsd, Med, Angle, Po, nh, Ph):
    epf= L-Q-(Lo-Log)/PD
    FRL= Er*Ar*((2/(rhor*Ar*Lr))**0.5)*((3.14/2*L)+Fr)
    Pm= (Med*Angle)+Po+(((1/nh)-1)*Ph-Po)*((Med*Angle/Ph)**2)
    res= (K1*epf)+(K2*FRL)+(K3*Mcsd)+(K4*Pm)
    return res

SPResult = SP (K1, K2, K3, K4, LL, Q, Lo, Log, PD, Er, Ar, rhor, Lr, Fr, Mcsd, Med, Angle, Po, nh, Ph)

if (SP (K1, K2, K3, K4, LL, Q, Lo, Log, PD, Er, Ar, rhor, Lr, Fr, Mcsd, Med, Angle, Po, nh, Ph)>=0):
    ans2 = "Pumping design is already optimum"
else :
    ans2 = "Pumping design is not optimum"

#Real Time Data
#Pumping Unit System
def PUS (load, min_load, max_load, PI, Pres, Pwf):
    #Basic Calculation
    AOF= PI*Pres
    Qo= PI*(Pres-Pwf)
    Opt_low= 0.9*0.8*AOF
    Opt_high= 1.1*0.8*AOF
    
    #Conditional
    if (load<min_load):
        hasil = "Swabbing with other wells, pumping load is under the minimum load"
    elif (load>max_load):
        hasil = "Swabbing with other wells, pumping load is more than the maximum load"
    elif (load>min_load & load<max_load & Qo>Opt_low & Qo<Opt_high):
        hasil = "Change running parameter or update SRP downhole equipment size"
    elif (load>max_load & Qo>Opt_low & Qo<Opt_high):
        hasil = "Upgrade pump unit"
    elif (Qo<Opt_low):
        hasil = "Check well productivity, production rate is under the optimum point"
    elif (Qo>Opt_high):
        hasil = "Check well productivity, production rate is more than the optimum point"
    return hasil

PUSResult = PUS (load, min_load, max_load, PI, Pres, Pwf)

#st.write("Structure",Structure)
#st.write("Well", Well)
#a = '''st.write("Pumping System Efficiency Value (%)",round(PSEResult,4))
#st.write("Remarks :", ans)
#st.write("Swabbing Parameter Analysis Value",round(SPResult,2))
#st.write("Remarks: ",ans2)
#st.write("Pumping Unit System Analysis Remarks :",PUSResult)
#'''
data = [[Structure,Well,round(PSEResult,4),ans,round(SPResult,2),ans2,PUSResult]]

output_df = pd.DataFrame(data,columns=["Structure","Well","Pumping System Efficiency Value (%)","Remarks","Swabbing Parameter Analysis Value","Remarks","Pumping Unit System Analysis Remarks"])

output_df=output_df.set_index('Structure')



st.subheader('**Input Data **')
st.write(df)



st.subheader('**Program Output**')
st.table(output_df)
