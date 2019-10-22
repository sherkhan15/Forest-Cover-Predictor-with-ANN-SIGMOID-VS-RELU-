#!/usr/bin/env python
# coding: utf-8

# # Information About Dataset

# This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections of forest. There are over half a million measurements total!

# ## **LOADING OF DATASET**
# This dataset includes information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography.

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset
import keras
from keras.utils import to_categorical
from keras import models
from keras import layers


# In[25]:


df=pd.read_csv('covtype.csv',index_col=0)


# In[26]:


import seaborn as sns


# In[27]:


import sklearn
from sklearn.preprocessing import StandardScaler


# In[28]:


df=pd.read_csv('covtype.csv',index_col=0)


# # Data Statistics

# In[29]:


df.head()


# In[ ]:





# In[30]:


df.info()
# We can see all columns have diffrent data types , float64(47), int64(7).


# *No Misssing values are presents*

# In[31]:


df.describe()


# **SHAPE**

# In[32]:


print(df.shape)

# We can see that there are 154340 instances having 55 attributes


# In[33]:


# Statistical description

pd.set_option('display.max_columns', None)
print(df.describe())

# Learning :
# No attribute is missing as count is 581012 for all attributes. Hence, all rows can be used
# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cant be used.
# Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis
# Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos


# **SKEWNESS**

# In[34]:


# Skewness of the distribution

print(df.skew())

# Values close to 0 show less skew
# Several attributes in Soil_Type show a large skew. Hence, some algos may benefit if skew is corrected


# 
# **Class Distribution**
# 
# 
# 

# In[35]:


# Number of instances belonging to each class

df.groupby('Cover_Type').size()


# We see that all classes not have an equal presence. So, class re-balancing is necessary


# In[36]:


a=df['Cover_Type']
sns.countplot(a)


# In[37]:


g = df.groupby('Cover_Type')
g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
g.head(8)


# # Explorartory Data Analysis

# In[38]:


import matplotlib.pyplot as plt


# **DATA INTERGRATION**
# 
# 
# 1. Correlation
# 
# 
# 

# * Correlation tells relation between two attributes.
# * Correlation requires continous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary

# In[39]:


import numpy



#sets the number of features considered
size = 10 

#create a dataframe with only 'size' features
data=df.iloc[:,:size] 

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Strong correlation is observed between the following pairs
# This represents an opportunity to reduce the feature set through transformations such as PCA


#   2) Scatter Plot(pairlot)
# 
# 
# * The plots show to which class does a point belong to. The class distribution overlaps in the plots.  
# 
# * Hillshade patterns give a nice ellipsoid patterns with each other
# 
# * Aspect and Hillshades attributes form a sigmoid pattern
# 
# * Horizontal and vertical distance to hydrology give an almost linear pattern. 

# In[40]:


for v,i,j in s_corr_list:
    sns.pairplot(df, hue="Cover_Type", height=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()


# # DATA VISUALIZATION

# 
# 
# *   HEAT MAP
# *   BOX PLOT
# *   PAIR PLOT
# 
# 

# In[41]:


col_list = df.columns
col_list = [col for col in col_list if not col[0:4]=='Soil']
fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(df[col_list].corr(),square=True,linewidths=1)
plt.title('Correlation of Variables')

plt.figure(figsize=(10,10))
sns.boxplot(y='Slope',x='Cover_Type', data= df )
plt.title('slope vs Cover_Type')


sns.pairplot( df, hue='Cover_Type',vars=['Aspect','Slope','Hillshade_9am','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Fire_Points'],diag_kind="kde")
plt.show()


# **LM PLOT**
# 
# *  Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrology with Soil_Type2
# *  Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrologywith Wilderness_Area1
# 
# 

# In[42]:



sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=df, hue='Soil_Type2',fit_reg=False)


# In[43]:


sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=df, hue='Wilderness_Area1',fit_reg=False)


# * **Violin Plot** - a combination of box and density plots

# In[44]:




#names of all the attributes 
cols = df.columns

#number of attributes (exclude target)
size = len(cols)-1

#x-axis has target attribute to distinguish between classes
x = cols[size]

#y-axis shows values of an attribute
y = cols[0:size]

#Plot violin for all attributes
for i in range(0,size):
    sns.violinplot(data=df,x=x,y=y[i])  
    plt.show()

#Elevation is has a separate distribution for most classes. Highly correlated with the target and hence an important attribute
#Aspect contains a couple of normal distribution for several classes
#Horizontal distance to road and hydrology have similar distribution
#Hillshade 9am and 12pm display left skew
#Hillshade 3pm is normal
#Lots of 0s in vertical distance to hydrology
#Wilderness_Area3 gives no class distinction. As values are not present, others gives some scope to distinguish
#Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes


# * Grouping of One hot encoded attributes
#     - Group one-hot encoded variables of a category into one single variable

# In[45]:


import pandas as pd
import numpy as np


# In[46]:




#names of all the columns
cols = df.shape
#number of rows=r , number of columns=c
r,c = df.shape


#Create a new dataframe with r rows, one column for each encoded category, and target in the end
data = pd.DataFrame(index=np.arange(0, r),columns=['Wilderness_Area','Soil_Type','Cover_Type'])

#Make an entry in 'data' for each r as category_id, target value
for i in range(0,r):
    w=0;
    s=0;
    # Category1 range
    for j in range(10,14):
        if (df.iloc[i,j] == 1):
            w=j-9  #category class
            break
    # Category2 range        
    for k in range(14,54):
        if (df.iloc[i,k] == 1):
            s=k-13 #category class
            break
    #Make an entry in 'data' for each r as category_id, target value        
    data.iloc[i]=[w,s,df.iloc[i,c-1]]

#Plot for Category1    
sns.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)
plt.show()
#Plot for Category2
plt.rc("figure", figsize=(25, 10))
sns.countplot(x="Soil_Type", hue="Cover_Type", data=data)
plt.show()


#WildernessArea_4 has a lot of presence for cover_type 4. Good class distinction
#WildernessArea_3 has not much class distinction
#SoilType 1-6,10-14,17, 22-23, 29-33,35,38-40 offer lot of class distinction as counts for some are very high


# # Data Cleaning 
# 
# *  Remove unnecessary columns

# In[47]:


#Removal list initialize
rem = []

#Add constant columns as they don't help in prediction process
for c in df.columns:
    if df[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
df.drop(rem,axis=1,inplace=True)

print(rem)


# # Normalizing DataSet

# In[48]:


from sklearn import preprocessing
df = pd.read_csv('covtype.csv')
x = df[df.columns[:55]]
y = df.Cover_Type
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)


# *Select numerical columns which needs to be normalized*

# In[49]:



train_norm = x_train[x_train.columns[0:10]]
test_norm = x_test[x_test.columns[0:10]]


# 
# *Normalize Training Data*
# 
# 

# In[50]:


std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)


# Converting numpy array to dataframe

# In[51]:


training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
print (x_train.head())


#  Normalize Testing Data by using mean and SD of training set

# In[52]:


x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_train.head())


# As y variable is multi class categorical variable, hence using softmax as activation function and sparse-categorical cross entropy as loss function.

# ***Validating Data Through Relu Function***

# In[53]:


model = keras.Sequential([
 keras.layers.Dense(64, activation=tf.nn.relu,                  
 input_shape=(x_train.shape[1],)),
 keras.layers.Dense(64, activation=tf.nn.relu),
 keras.layers.Dense(8, activation=  'softmax')
 ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history2 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))


# **Visualize Training History**

# In[54]:



from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
history = model.fit(x_train, y_train, nb_epoch=26, validation_split=0.7, shuffle=True)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# **TanH**

# In[55]:


X = df.drop(['Cover_Type'],axis=1)
Y = df['Cover_Type']


# In[56]:


from keras.models import Sequential
from keras.layers import Dense,Dropout 
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# In[79]:


model = Sequential()
model.add(Dense(100,input_dim=54,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(25,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(25,activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
model.fit(X,Y,validation_split=.33,epochs=10,batch_size=100,verbose=1,callbacks=[reduce_lr,early])


# In[80]:


def on_train_begin(self, logs=None):
    self.epoch = []
    self.history = {}
def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epoch.append(epoch)
    for k, v in logs.items():
        self.history.setdefault(k, []).append(v)


# In[84]:



record = model.fit(X,Y,validation_split=.33,epochs=5,batch_size=100,verbose=1)


# In[85]:


record.epoch


# In[86]:


record.history


# In[89]:


history_dict = record.history
train_acc = history_dict['loss']
val_acc = history_dict['val_loss']
epochs = range(1, len(history_dict['loss'])+1)
plt.plot(epochs, train_acc,'bo',label='Training Accuracy')
plt.plot(epochs, val_acc,'b',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[90]:



import matplotlib.pyplot as plt
import numpy
history = model.fit(X,Y,validation_split=.33,epochs=20,batch_size=100,verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # sigmoid

# In[2]:


import numpy as np


# In[5]:


df = pd.read_csv('covtype.csv')


# In[6]:


df_list = df[['Aspect',                 
'Slope',                               
'Horizontal_Distance_To_Hydrology' ,   
'Vertical_Distance_To_Hydrology',        
'Horizontal_Distance_To_Roadways',       
'Hillshade_9am' ,                     
'Hillshade_Noon',                        
'Hillshade_3pm' ,                        
'Horizontal_Distance_To_Fire_Points',   
'Wilderness_Area1' ,                     
'Wilderness_Area2',                      
'Wilderness_Area3' ,                     
'Wilderness_Area4' ,                    
'Soil_Type1'  ,                        
'Soil_Type2',                            
'Soil_Type3' ,                           
'Soil_Type4' ,                          
'Soil_Type5',                            
'Soil_Type6' ,                          
'Soil_Type7' ,                           
'Soil_Type8'  ,                          
'Soil_Type9'  ,                          
'Soil_Type10' ,                          
'Soil_Type11' ,                          
'Soil_Type12' ,                          
'Soil_Type13',                           
'Soil_Type14' ,                         
'Soil_Type15' ,                          
'Soil_Type16' ,                         
'Soil_Type17' ,                          
'Soil_Type18' ,                         
'Soil_Type19' ,                          
'Soil_Type20',                           
'Soil_Type21',                          
'Soil_Type22',                           
'Soil_Type23',                          
'Soil_Type24',                           
'Soil_Type25',                          
'Soil_Type26' ,                         
'Soil_Type27',                           
'Soil_Type28',                           
'Soil_Type29' ,                          
'Soil_Type30',                         
'Soil_Type31' ,                          
'Soil_Type32',                           
'Soil_Type33',                         
'Soil_Type34',                        
'Soil_Type35' ,                         
'Soil_Type36' ,                          
'Soil_Type37' ,                          
'Soil_Type38' ,                       
'Soil_Type39' ,                       
'Soil_Type40']]
df_list1 = df[['Cover_Type']]


# In[7]:


X = np.array((df_list), dtype=float)
y = np.array((df_list1), dtype=float)


# In[8]:


X = X/np.max(X,axis=0)
y = y/np.max(y,axis=0)


# In[ ]:


class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 53
        self.outputSize = 1
        self.hiddenSize = 45
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        
    def feedForward(self, X):
        self.z = np.dot(X, self.W1) 
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.sigmoid(self.z3)
        return output
        
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) 
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) 
        
        self.W1 += X.T.dot(self.z2_delta) 
        self.W2 += self.z2.T.dot(self.output_delta)
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()

for i in range(1000):
    NN.train(X, y)
        
print(NN.feedForward(X[0:20]))


# In[ ]:




