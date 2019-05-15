# Databricks notebook source
# MAGIC %md #Predicting Flight Delay

# COMMAND ----------

# MAGIC %md ##Objective
# MAGIC 
# MAGIC In this project, we are predicting the delay in departure of flights.The goal of the project is to perform analysis on the flight data to gain valuable insights and build models to predict by how many minutes the flight departure will be delayed, given a set of characteristics. We have used Linear Regression model, Random Forest Regressor model and Decison Tree Regressor model to find the accuracy of our prediction.

# COMMAND ----------

# MAGIC %md ##Dataset Description
# MAGIC 
# MAGIC The dataset was obtained from the Bureau of Transportation Statistics for January 2019.The dataset consist of 583985 records.The link is https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236
# MAGIC 
# MAGIC The data fields that were selected are explained below:
# MAGIC 
# MAGIC ** SYS_FIELD_NAME	 FIELD_DESC **
# MAGIC * YEAR	Year
# MAGIC * MONTH	Month
# MAGIC * DAY_OF_MONTH	Day of Month
# MAGIC * DAY_OF_WEEK	Day of Week
# MAGIC * FL_DATE	Flight Date (yyyymmdd)
# MAGIC * OP_UNIQUE_CARRIER	Unique Carrier Code. When the same code has been used by multiple carriers, a numeric suffix is used for earlier users, for example, PA, PA(1), PA(2). Use this field for analysis across a range of years.
# MAGIC * OP_CARRIER_FL_NUM	Flight Number
# MAGIC * ORIGIN_AIRPORT_ID	Origin Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport.  Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC * ORIGIN_AIRPORT_SEQ_ID	Origin Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time.  Airport attributes, such as airport name or coordinates, may change over time.
# MAGIC * ORIGIN_CITY_MARKET_ID	Origin Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market.  Use this field to consolidate airports serving the same city market.
# MAGIC * ORIGIN_CITY_NAME	Origin Airport, City Name
# MAGIC * DEST_AIRPORT_ID	Destination Airport, Airport ID. An identification number assigned by US DOT to identify a unique airport.  Use this field for airport analysis across a range of years because an airport can change its airport code and airport codes can be reused.
# MAGIC * DEST_AIRPORT_SEQ_ID	Destination Airport, Airport Sequence ID. An identification number assigned by US DOT to identify a unique airport at a given point of time.  Airport attributes, such as airport name or coordinates, may change over time.
# MAGIC * DEST_CITY_MARKET_ID	Destination Airport, City Market ID. City Market ID is an identification number assigned by US DOT to identify a city market.  Use this field to consolidate airports serving the same city market.
# MAGIC * DEST_CITY_NAME	Destination Airport, City Name
# MAGIC * DEP_TIME	Actual Departure Time (local time: hhmm)
# MAGIC * DEP_DELAY_NEW	Difference in minutes between scheduled and actual departure time. Early departures set to 0.
# MAGIC * DEP_DEL15	Departure Delay Indicator, 15 Minutes or More (1=Yes)
# MAGIC * ARR_TIME	Actual Arrival Time (local time: hhmm)
# MAGIC * ARR_DELAY_NEW	Difference in minutes between scheduled and actual arrival time. Early arrivals set to 0.
# MAGIC * ARR_DEL15	Arrival Delay Indicator, 15 Minutes or More (1=Yes)
# MAGIC * AIR_TIME	Flight Time, in Minutes
# MAGIC * FLIGHTS	Number of Flights
# MAGIC * DISTANCE	Distance between airports (miles)
# MAGIC * CARRIER_DELAY	Carrier Delay, in Minutes
# MAGIC * WEATHER_DELAY	Weather Delay, in Minutes
# MAGIC * NAS_DELAY	National Air System Delay, in Minutes
# MAGIC * SECURITY_DELAY	Security Delay, in Minutes
# MAGIC * LATE_AIRCRAFT_DELAY	Late Aircraft Delay, in Minutes
# MAGIC 
# MAGIC 
# MAGIC Since we are going to predict the departure delay of the flight in minutes, our predictor field is DEP_DELAY_NEW. It is defined as the difference in minutes between scheduled and actual departure time. For, early departures or on-time flights, this field is set to 0.

# COMMAND ----------

# MAGIC %md ## To Download the data
# MAGIC 
# MAGIC Since we didnt get the direct download link from the website, we used curl -L --location Follow redirects and -d -data <data> Send specified data in POST request to download the required csv file.The link is https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L -d "UserTableName=Reporting_Carrier_On_Time_Performance_1987_present&DBShortName=&RawDataTable=T_ONTIME_REPORTING&sqlstr=+SELECT+YEAR%2CMONTH%2CDAY_OF_MONTH%2CDAY_OF_WEEK%2CFL_DATE%2COP_UNIQUE_CARRIER%2COP_CARRIER_AIRLINE_ID%2COP_CARRIER_FL_NUM%2CORIGIN_AIRPORT_ID%2CORIGIN_AIRPORT_SEQ_ID%2CORIGIN_CITY_MARKET_ID%2CORIGIN%2CORIGIN_CITY_NAME%2CORIGIN_STATE_ABR%2CDEST_AIRPORT_ID%2CDEST_AIRPORT_SEQ_ID%2CDEST_CITY_MARKET_ID%2CDEST%2CDEST_CITY_NAME%2CDEST_STATE_ABR%2CDEP_TIME%2CDEP_DELAY%2CDEP_DELAY_NEW%2CDEP_DEL15%2CARR_TIME%2CARR_DELAY%2CARR_DEL15%2CAIR_TIME%2CFLIGHTS%2CDISTANCE%2CCARRIER_DELAY%2CWEATHER_DELAY%2CNAS_DELAY%2CSECURITY_DELAY%2CLATE_AIRCRAFT_DELAY+FROM++T_ONTIME_REPORTING+WHERE+Month+%3D1+AND+YEAR%3D2019&varlist=YEAR%2CMONTH%2CDAY_OF_MONTH%2CDAY_OF_WEEK%2CFL_DATE%2COP_UNIQUE_CARRIER%2COP_CARRIER_AIRLINE_ID%2COP_CARRIER_FL_NUM%2CORIGIN_AIRPORT_ID%2CORIGIN_AIRPORT_SEQ_ID%2CORIGIN_CITY_MARKET_ID%2CORIGIN%2CORIGIN_CITY_NAME%2CORIGIN_STATE_ABR%2CDEST_AIRPORT_ID%2CDEST_AIRPORT_SEQ_ID%2CDEST_CITY_MARKET_ID%2CDEST%2CDEST_CITY_NAME%2CDEST_STATE_ABR%2CDEP_TIME%2CDEP_DELAY%2CDEP_DELAY_NEW%2CDEP_DEL15%2CARR_TIME%2CARR_DELAY%2CARR_DEL15%2CAIR_TIME%2CFLIGHTS%2CDISTANCE%2CCARRIER_DELAY%2CWEATHER_DELAY%2CNAS_DELAY%2CSECURITY_DELAY%2CLATE_AIRCRAFT_DELAY&grouplist=&suml=&sumRegion=&filter1=title%3D&filter2=title%3D&geo=All%A0&time=January&timename=Month&GEOGRAPHY=All&XYEAR=2019&FREQUENCY=1&VarName=YEAR&VarDesc=Year&VarType=Num&VarDesc=Quarter&VarType=Num&VarName=MONTH&VarDesc=Month&VarType=Num&VarName=DAY_OF_MONTH&VarDesc=DayofMonth&VarType=Num&VarName=DAY_OF_WEEK&VarDesc=DayOfWeek&VarType=Num&VarName=FL_DATE&VarDesc=FlightDate&VarType=Char&VarName=OP_UNIQUE_CARRIER&VarDesc=Reporting_Airline&VarType=Char&VarName=OP_CARRIER_AIRLINE_ID&VarDesc=DOT_ID_Reporting_Airline&VarType=Num&VarDesc=IATA_CODE_Reporting_Airline&VarType=Char&VarDesc=Tail_Number&VarType=Char&VarName=OP_CARRIER_FL_NUM&VarDesc=Flight_Number_Reporting_Airline&VarType=Char&VarName=ORIGIN_AIRPORT_ID&VarDesc=OriginAirportID&VarType=Num&VarName=ORIGIN_AIRPORT_SEQ_ID&VarDesc=OriginAirportSeqID&VarType=Num&VarName=ORIGIN_CITY_MARKET_ID&VarDesc=OriginCityMarketID&VarType=Num&VarName=ORIGIN&VarDesc=Origin&VarType=Char&VarName=ORIGIN_CITY_NAME&VarDesc=OriginCityName&VarType=Char&VarName=ORIGIN_STATE_ABR&VarDesc=OriginState&VarType=Char&VarDesc=OriginStateFips&VarType=Char&VarDesc=OriginStateName&VarType=Char&VarDesc=OriginWac&VarType=Num&VarName=DEST_AIRPORT_ID&VarDesc=DestAirportID&VarType=Num&VarName=DEST_AIRPORT_SEQ_ID&VarDesc=DestAirportSeqID&VarType=Num&VarName=DEST_CITY_MARKET_ID&VarDesc=DestCityMarketID&VarType=Num&VarName=DEST&VarDesc=Dest&VarType=Char&VarName=DEST_CITY_NAME&VarDesc=DestCityName&VarType=Char&VarName=DEST_STATE_ABR&VarDesc=DestState&VarType=Char&VarDesc=DestStateFips&VarType=Char&VarDesc=DestStateName&VarType=Char&VarDesc=DestWac&VarType=Num&VarDesc=CRSDepTime&VarType=Char&VarName=DEP_TIME&VarDesc=DepTime&VarType=Char&VarName=DEP_DELAY&VarDesc=DepDelay&VarType=Num&VarName=DEP_DELAY_NEW&VarDesc=DepDelayMinutes&VarType=Num&VarName=DEP_DEL15&VarDesc=DepDel15&VarType=Num&VarDesc=DepartureDelayGroups&VarType=Num&VarDesc=DepTimeBlk&VarType=Char&VarDesc=TaxiOut&VarType=Num&VarDesc=WheelsOff&VarType=Char&VarDesc=WheelsOn&VarType=Char&VarDesc=TaxiIn&VarType=Num&VarDesc=CRSArrTime&VarType=Char&VarName=ARR_TIME&VarDesc=ArrTime&VarType=Char&VarName=ARR_DELAY&VarDesc=ArrDelay&VarType=Num&VarDesc=ArrDelayMinutes&VarType=Num&VarName=ARR_DEL15&VarDesc=ArrDel15&VarType=Num&VarDesc=ArrivalDelayGroups&VarType=Num&VarDesc=ArrTimeBlk&VarType=Char&VarDesc=Cancelled&VarType=Num&VarDesc=CancellationCode&VarType=Char&VarDesc=Diverted&VarType=Num&VarDesc=CRSElapsedTime&VarType=Num&VarDesc=ActualElapsedTime&VarType=Num&VarName=AIR_TIME&VarDesc=AirTime&VarType=Num&VarName=FLIGHTS&VarDesc=Flights&VarType=Num&VarName=DISTANCE&VarDesc=Distance&VarType=Num&VarDesc=DistanceGroup&VarType=Num&VarName=CARRIER_DELAY&VarDesc=CarrierDelay&VarType=Num&VarName=WEATHER_DELAY&VarDesc=WeatherDelay&VarType=Num&VarName=NAS_DELAY&VarDesc=NASDelay&VarType=Num&VarName=SECURITY_DELAY&VarDesc=SecurityDelay&VarType=Num&VarName=LATE_AIRCRAFT_DELAY&VarDesc=LateAircraftDelay&VarType=Num&VarDesc=FirstDepTime&VarType=Char&VarDesc=TotalAddGTime&VarType=Num&VarDesc=LongestAddGTime&VarType=Num&VarDesc=DivAirportLandings&VarType=Num&VarDesc=DivReachedDest&VarType=Num&VarDesc=DivActualElapsedTime&VarType=Num&VarDesc=DivArrDelay&VarType=Num&VarDesc=DivDistance&VarType=Num&VarDesc=Div1Airport&VarType=Char&VarDesc=Div1AirportID&VarType=Num&VarDesc=Div1AirportSeqID&VarType=Num&VarDesc=Div1WheelsOn&VarType=Char&VarDesc=Div1TotalGTime&VarType=Num&VarDesc=Div1LongestGTime&VarType=Num&VarDesc=Div1WheelsOff&VarType=Char&VarDesc=Div1TailNum&VarType=Char&VarDesc=Div2Airport&VarType=Char&VarDesc=Div2AirportID&VarType=Num&VarDesc=Div2AirportSeqID&VarType=Num&VarDesc=Div2WheelsOn&VarType=Char&VarDesc=Div2TotalGTime&VarType=Num&VarDesc=Div2LongestGTime&VarType=Num&VarDesc=Div2WheelsOff&VarType=Char&VarDesc=Div2TailNum&VarType=Char&VarDesc=Div3Airport&VarType=Char&VarDesc=Div3AirportID&VarType=Num&VarDesc=Div3AirportSeqID&VarType=Num&VarDesc=Div3WheelsOn&VarType=Char&VarDesc=Div3TotalGTime&VarType=Num&VarDesc=Div3LongestGTime&VarType=Num&VarDesc=Div3WheelsOff&VarType=Char&VarDesc=Div3TailNum&VarType=Char&VarDesc=Div4Airport&VarType=Char&VarDesc=Div4AirportID&VarType=Num&VarDesc=Div4AirportSeqID&VarType=Num&VarDesc=Div4WheelsOn&VarType=Char&VarDesc=Div4TotalGTime&VarType=Num&VarDesc=Div4LongestGTime&VarType=Num&VarDesc=Div4WheelsOff&VarType=Char&VarDesc=Div4TailNum&VarType=Char&VarDesc=Div5Airport&VarType=Char&VarDesc=Div5AirportID&VarType=Num&VarDesc=Div5AirportSeqID&VarType=Num&VarDesc=Div5WheelsOn&VarType=Char&VarDesc=Div5TotalGTime&VarType=Num&VarDesc=Div5LongestGTime&VarType=Num&VarDesc=Div5WheelsOff&VarType=Char&VarDesc=Div5TailNum&VarType=Char" "https://www.transtats.bts.gov/DownLoad_Table.asp?Table_ID=236&Has_Group=3&Is_Zipped=0" > m1.zip

# COMMAND ----------

# MAGIC %md ## To remove any old csv file which gets generated for each download and to unzip m1 folder

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf m1
# MAGIC unzip m1.zip -d m1
# MAGIC ls m1/*.csv | head -1 | xargs -i@ mv @ m1project.csv

# COMMAND ----------

# MAGIC %md ###To read the csv file and store it in a dataframe

# COMMAND ----------

df = spark.read.csv(path='file:///databricks/driver/m1project.csv',header='true', inferSchema ='true', sep=',', mode='DROPMALFORMED')

# COMMAND ----------

display(df.limit(50))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md ## Data Cleaning 
# MAGIC * We have cleaned the data by removing the null values for the required fields. No other cleaning was necessary for our dataset.

# COMMAND ----------

print("Number of records: " + str(df.count()))
df = df.na.drop(subset=["CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"])
df=df.drop('_c29')
print("Number of records after cleaning: " + str(df.count()))

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md ### Display a summary of the stats: count, mean, stddev, min, etc.

# COMMAND ----------


display(df.describe())

# COMMAND ----------

# MAGIC %md ###Removing the columns which are not required for prediction

# COMMAND ----------

df1 = df.drop('YEAR', 'MONTH', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID','ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEP_TIME', 'ARR_DELAY_NEW', 'DEP_DEL15','ARR_DEL15','ARR_TIME', 'FLIGHTS','FL_DATE','ORIGIN_CITY_NAME','DEST_CITY_NAME','ORIGIN','ORIGIN_STATE_ABR','DEST','DEST_CITY_NAME','DEST_STATE_ABR','DEP_DELAY','_c35','OP_CARRIER_AIRLINE_ID','ARR_DELAY')

# COMMAND ----------

display(df1.limit(10))

# COMMAND ----------

# MAGIC %md ### Casting the columns 

# COMMAND ----------

df1 = df1.withColumn('DAY_OF_MONTH',df1['DAY_OF_MONTH'].cast("double"))  
df1 = df1.withColumn('DAY_OF_WEEK',df1['DAY_OF_WEEK'].cast("double"))
df1.printSchema()

# COMMAND ----------

# MAGIC %md ##Finding the correlation using heatmap

# COMMAND ----------

import matplotlib.pyplot as plt 
import seaborn as sns 

pdf = df1.toPandas()
fig, ax = plt.subplots()
fig.set_size_inches(30, 20)
corr = pdf.corr()
sns.heatmap(corr)
display(fig)

# COMMAND ----------

# MAGIC %md ##Split the data set for training and testing

# COMMAND ----------

train_data, test_data  = df1.randomSplit([0.6, 0.4], 24)   # proportions [], seed for random

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))


# COMMAND ----------

display(train_data.limit(5))

# COMMAND ----------

# MAGIC %md ##The old way to do this...

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler,RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline, Model
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

columns = df1.columns
# Not using DEP_DELAY_NEW (label) 
columns.remove('DEP_DELAY_NEW')


# with RFormula
#Predictor field is DEP_DELAY_NEW
formula = "{} ~ {}".format("DEP_DELAY_NEW", " + ".join(columns))
print("Formula : {}".format(formula))
rformula = RFormula(formula = formula)
lr = LinearRegression(labelCol ="label", featuresCol ="features")
# The ML pipeline
pipeline = Pipeline(stages=[rformula, lr])
fittedPipe = pipeline.fit(train_data)

# COMMAND ----------

fittedPipe = pipeline.fit(train_data)
fittedPipe.stages
lrModel1 = fittedPipe.stages[1]

# COMMAND ----------

# MAGIC %md ## Data Modelling using 3 models
# MAGIC *  Efficiently run multiple pipelines in parameter grid
# MAGIC 
# MAGIC Models Used:
# MAGIC 
# MAGIC * Linear Regression
# MAGIC * Decision Tree
# MAGIC * Random Forest
# MAGIC 
# MAGIC 
# MAGIC Param Grid and Cross Validation are performed over the models

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler,RFormula
from pyspark.ml.regression import LinearRegression , GeneralizedLinearRegression, DecisionTreeRegressor,RandomForestRegressor
from pyspark.ml import Pipeline, Model
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

columns = df1.columns
# Not using DEP_DELAY_NEW (label) 
columns.remove('DEP_DELAY_NEW')

# with RFormula
#Predictor field is DEP_DELAY_NEW
formula = "{} ~ {}".format("DEP_DELAY_NEW", " + ".join(columns))
print("Formula : {}".format(formula))
rformula = RFormula(formula = formula)
# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[])  # Must initialize with empty list!

# base pipeline (the processing here should be reused across pipelines)
basePipeline =[rformula]

#############################################################
# Specify Linear Regression model
lr = LinearRegression()
pl_lr = basePipeline + [lr]
pg_lr = ParamGridBuilder()\
          .baseOn({pipeline.stages: pl_lr})\
          .addGrid(lr.regParam,[0.01, .04])\
          .build()


#############################################################
# Specify Decision Tree model
dt = DecisionTreeRegressor()
pl_dt = basePipeline + [dt]
pg_dt = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_dt})\
      .build()
#############################################################
# Specify Random Forest  model
rff = RandomForestRegressor()
pl_rff = basePipeline + [rff]
pg_rff = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_rff})\
      .build()

# One grid from the individual grids
paramGrid = pg_lr + pg_dt + pg_rff

# COMMAND ----------

# The regression metric can be  rmse, r2
# See the metrics here https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#regression-model-evaluation
# Should run more than 3 folds, but here we simplify so that it will complete
cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(RegressionEvaluator()\
                       .setMetricName("r2"))\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(3)

cvModel = cv.fit(df1)

# COMMAND ----------

# MAGIC %md ## Best and Worst Model

# COMMAND ----------

import numpy as np
# RegressionEvaluator metric name is r2, so higher is better
# http://gim.unmc.edu/dxtests/roc3.htm
print("Best Model")
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Worst Model")
print (cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ])

# COMMAND ----------

# MAGIC %md ##Data Transformation and Prediction from model
# MAGIC * The model is transformed and predictions from the models are made

# COMMAND ----------

## Make predictions on test documents. 
train_data, test_data  = df.randomSplit([0.6, 0.4], 24)   # proportions [], seed for random
# CrossValidator.fit() is in cvModel, which is the best model found.
predictions = cvModel.transform(test_data)
display(predictions.select('label', 'prediction').limit(100))
labeledprediction = cvModel.transform(test_data).select("label" , "prediction")

# COMMAND ----------

# MAGIC %md ## To show the predicted results using scatter-plot for the best model

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
y_test,predictions = zip(*labeledprediction.collect())
fig, ax = plt.subplots()
plt.scatter(y_test,predictions)
display(fig)

# COMMAND ----------

# MAGIC %md ##Prediction Summary
# MAGIC 
# MAGIC We are predicting the number of minutes the flight is been delayed based on DAY_OF_MONTH,DAY_OF_WEEK,OP_UNIQUE_CARRIER,AIR_TIME,DISTANCE , CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Carrier delay is within the control of the air carrier. Examples of occurrences that may determine carrier delay are: aircraft cleaning, aircraft damage, awaiting the arrival of connecting passengers or crew, baggage, bird strike, cargo loading, catering, computer, outage-carrier equipment, crew legality (pilot or attendant rest), damage by hazardous goods, engineering inspection, fueling, handling disabled passengers, late crew, lavatory servicing, maintenance, oversales, potable water servicing, removal of unruly passenger, slow boarding or seating, stowing carry-on baggage, weight and balance delays.
# MAGIC 
# MAGIC 
# MAGIC Arrival delay at an airport due to the late arrival of the same aircraft at a previous airport. The ripple effect of an earlier delay at downstream airports is referred to as delay propagation.
# MAGIC 
# MAGIC 
# MAGIC Delay that is within the control of the National Airspace System (NAS) may include: non-extreme weather conditions, airport operations, heavy traffic volume, air traffic control, etc. Delays that occur after Actual Gate Out are usually attributed to the NAS and are also reported through OPSNET.
# MAGIC 
# MAGIC 
# MAGIC Security delay is caused by evacuation of a terminal or concourse, re-boarding of aircraft because of security breach, inoperative screening equipment and/or long lines in excess of 29 minutes at screening areas.
# MAGIC 
# MAGIC 
# MAGIC Weather delay is caused by extreme or hazardous weather conditions that are forecasted or manifest themselves on point of departure, enroute, or on point of arrival.

# COMMAND ----------

# MAGIC %md ## Model Evaluation
# MAGIC * Model evaluations are done and then compared to find the Best and the worst performing model

# COMMAND ----------

# Summarize the model over the training set and print out some metrics
print("Best pipeline", cvModel.bestModel.stages)
print("Best model", cvModel.bestModel.stages[1])


# COMMAND ----------

trainingSummary = lrModel1.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

lr = cvModel.bestModel.stages[1]
cvModel.bestModel.stages[1]
testsummary = lr.summary
print("RMSE: %f" % testsummary.rootMeanSquaredError)
print("r2: %f" % testsummary.r2)

# COMMAND ----------

# MAGIC %md ##Model measures for plot

# COMMAND ----------

import re
def paramGrid_model_name(model):
  params = [v for v in model.values() if type(v) is not list]
  name = [v[-1] for v in model.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return "{}{}".format(name,params)

# Resulting metric and model description
# get the measure from the CrossValidator, cvModel.avgMetrics
# get the model name & params from the paramGrid
# put them together here:
measures = zip(cvModel.avgMetrics, [paramGrid_model_name(m) for m in paramGrid])
metrics,model_names = zip(*measures)

# COMMAND ----------

# MAGIC %md ##Visualization of Model evaluation

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf() # clear figure
fig = plt.figure( figsize=(5, 5))
plt.style.use('fivethirtyeight')
axis = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# plot the metrics as Y
#plt.plot(range(len(model_names)),metrics)
plt.bar(range(len(model_names)),metrics)
# plot the model name & param as X labels
plt.xticks(range(len(model_names)), model_names, rotation=70, fontsize=6)
plt.yticks(fontsize=6)
#plt.xlabel('model',fontsize=8)
plt.ylabel('R2 (higher is better)',fontsize=8)
plt.title('Model evaluations')
display(plt.show())

# COMMAND ----------

# MAGIC %md ##Results Explained
# MAGIC 
# MAGIC * Evaluation Summary
# MAGIC 
# MAGIC We have evaluated our models based on the values of r-square . R-square is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.
# MAGIC R2 value should be between 0 and 1.
# MAGIC '0' indicates that the model explains none of the variability of the response data around its mean and 1  indicates that the model explains all the variability of the response data around its mean.
# MAGIC The higher the R-squared, the better the model fits your data
# MAGIC 
# MAGIC The root-mean-squared error (RMSE) is a measure of how well the model performed. It does this by measuring difference between predicted values and the actual values.In a good model, the RMSE should be close for both  testing data and training data. If the RMSE for testing data is higher than the training data, there is a high chance that your model overfit. In other words, your model performed worse during testing than training.
# MAGIC 
# MAGIC ###With the results of the evaluation, we observed that the Linear regression model is the best with the R2 value .96 ,while the random forest regressor is the worst model.Also we see that RMSE of the testing is less than the training data which ruled out overfitting and proves that our model is good.

# COMMAND ----------

# MAGIC %md ## Data Visualization

# COMMAND ----------

# MAGIC %md ##Top 10 carriers causing the maximum delay in departure
# MAGIC 
# MAGIC 
# MAGIC * OO-Skywest Airlines Inc.
# MAGIC * WN-Southwest Airlines Co.
# MAGIC * AA-American Airlines Inc.
# MAGIC * UA-United Air Lines Inc.
# MAGIC * DL-Delta Air Lines Inc.
# MAGIC * B6-JetBlue Airways
# MAGIC * YX-Republic Airline
# MAGIC * 9E-Endeavor Air
# MAGIC * MQ-American Eagle Airlines Inc.
# MAGIC * EV-Atlantic Southeast Airlines

# COMMAND ----------

from pyspark.sql.functions import desc
display(df1
  .select("OP_UNIQUE_CARRIER", "DEP_DELAY_NEW")
  .groupBy("OP_UNIQUE_CARRIER")
  .sum()
  .sort(desc("sum(DEP_DELAY_NEW)"))
  .limit(10))

# COMMAND ----------

# MAGIC %md ##Graph Frame
# MAGIC 
# MAGIC Import graph frame library

# COMMAND ----------

df2 = df.select("DAY_OF_WEEK","FL_DATE","OP_UNIQUE_CARRIER","ORIGIN_AIRPORT_ID","ORIGIN","ORIGIN_STATE_ABR","DEST_AIRPORT_ID","DEST","DEST_STATE_ABR","ARR_DELAY","AIR_TIME","DISTANCE","CARRIER_DELAY","NAS_DELAY","WEATHER_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","ORIGIN_CITY_NAME","DEST_CITY_NAME","DEP_DELAY_NEW")
df2.registerTempTable("df2")

# COMMAND ----------

from pyspark.sql.functions import split
split_col = split(df2['ORIGIN_CITY_NAME'], ',')
df2 = df2.withColumn('city_src', split_col.getItem(0))

from pyspark.sql.functions import split
split_col = split(df2['DEST_CITY_NAME'], ',')
df2 = df2.withColumn('city_dst', split_col.getItem(0))

# COMMAND ----------

df2 = df2.drop("ORIGIN_CITY_NAME", "DEST_CITY_NAME")

# COMMAND ----------

display(df2)

# COMMAND ----------

# MAGIC %md ## Renaming the columns of ORIGIN and DEST to src and dst for graph frames

# COMMAND ----------

df2 = df2.withColumnRenamed("ORIGIN","src")
df2 = df2.withColumnRenamed("DEST","dst")
df2 = df2.withColumnRenamed("ORIGIN_STATE_ABR","state_src")
df2 = df2.withColumnRenamed("DEST_STATE_ABR","state_dst")

# COMMAND ----------

from pyspark.sql.functions import lit
df2 = df2.withColumn("Country", lit("USA"))

# COMMAND ----------

df3 = df2.select("src", "city_src", "state_src", "Country").distinct()
df3 = df3.withColumnRenamed("city_src","City")
df3 = df3.withColumnRenamed("state_src","State")

# COMMAND ----------

display(df3)

# COMMAND ----------

# MAGIC %md ## Creating Vertices and Edges for graphframes
# MAGIC 
# MAGIC Attach graphframe library to your cluster

# COMMAND ----------


from pyspark.sql.functions import *
from graphframes import GraphFrame

#Create Vertices (airports) and Edges (flights)
tripVertices = df3.withColumnRenamed("src", "id").distinct()
df2 = df2.withColumnRenamed("DEP_DELAY_NEW","delay")
tripEdges = df2.select("delay", "src", "dst", "city_dst", "state_dst")
tripEdges.registerTempTable("tripEdges")
#Cache Vertices and Edges
tripEdges.cache()
tripVertices.cache()

# COMMAND ----------

display(tripVertices)

# COMMAND ----------

display(tripEdges)

# COMMAND ----------

# Build `tripGraph` GraphFrame
# This GraphFrame builds up on the vertices and edges based on our trips (flights)
tripGraph = GraphFrame(tripVertices, tripEdges)
tripGraph.cache()

#Build `tripGraphPrime` GraphFrame
#This graphframe contains a smaller subset of data to make it easier to display motifs and subgraphs (below)
tripEdgesPrime = df2.select("delay", "src", "dst")
tripGraphPrime = GraphFrame(tripVertices, tripEdgesPrime)

# COMMAND ----------

tripGraph.vertices.count()

# COMMAND ----------

tripGraph.edges.count()

# COMMAND ----------

# MAGIC %md ##What destinations tend to have significant delays for flights departing from San Fransisco

# COMMAND ----------

display(tripGraph.edges.filter("src = 'SFO' and delay > 0"))

# COMMAND ----------

# MAGIC %md ##What flights departing from Atlanta are most likely to have significant delays

# COMMAND ----------

display(tripGraph.edges.filter("src = 'ATL' and delay > 0").groupBy("src", "dst").avg("delay").sort(desc("avg(delay)")))

# COMMAND ----------

tripDelays = tripGraph.edges.where("DEP_DELAY > 0")
display(tripDelays)

# COMMAND ----------

# MAGIC %md ## To determine the most popular flights (single city hops)

# COMMAND ----------

# Determine the most popular flights (single city hops)
import pyspark.sql.functions as func
topTrips = tripGraph \
  .edges \
  .groupBy("src", "dst") \
  .agg(func.count("delay").alias("trips"))
display(topTrips.orderBy(topTrips.trips.desc()).limit(20))

# COMMAND ----------

# MAGIC %md ## To determining Airport ranking of importance using `pageRank`

# COMMAND ----------

# Determining Airport ranking of importance using `pageRank`
ranks = tripGraph.pageRank(resetProbability=0.15, maxIter=5)
display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(20))

# COMMAND ----------

# MAGIC %md ##D3 Visualization

# COMMAND ----------

# MAGIC %scala
# MAGIC package d3a
# MAGIC // We use a package object so that we can define top level classes like Edge that need to be used in other cells
# MAGIC 
# MAGIC import org.apache.spark.sql._
# MAGIC import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML
# MAGIC 
# MAGIC case class Edge(src: String, dest: String, count: Long)
# MAGIC 
# MAGIC case class Node(name: String)
# MAGIC case class Link(source: Int, target: Int, value: Long)
# MAGIC case class Graph(nodes: Seq[Node], links: Seq[Link])
# MAGIC 
# MAGIC object graphs {
# MAGIC val sqlContext = SQLContext.getOrCreate(org.apache.spark.SparkContext.getOrCreate())
# MAGIC import sqlContext.implicits._
# MAGIC 
# MAGIC def force(clicks: Dataset[Edge], height: Int = 100, width: Int = 960): Unit = {
# MAGIC   val data = clicks.collect()
# MAGIC   val nodes = (data.map(_.src) ++ data.map(_.dest)).map(_.replaceAll("_", " ")).toSet.toSeq.map(Node)
# MAGIC   val links = data.map { t =>
# MAGIC     Link(nodes.indexWhere(_.name == t.src.replaceAll("_", " ")), nodes.indexWhere(_.name == t.dest.replaceAll("_", " ")), t.count / 20 + 1)
# MAGIC   }
# MAGIC   showGraph(height, width, Seq(Graph(nodes, links)).toDF().toJSON.first())
# MAGIC }
# MAGIC 
# MAGIC /**
# MAGIC  * Displays a force directed graph using d3
# MAGIC  * input: {"nodes": [{"name": "..."}], "links": [{"source": 1, "target": 2, "value": 0}]}
# MAGIC  */
# MAGIC def showGraph(height: Int, width: Int, graph: String): Unit = {
# MAGIC 
# MAGIC displayHTML(s"""<!DOCTYPE html>
# MAGIC <html>
# MAGIC   <head>
# MAGIC     <link type="text/css" rel="stylesheet" href="https://mbostock.github.io/d3/talk/20111116/style.css"/>
# MAGIC     <style type="text/css">
# MAGIC       #states path {
# MAGIC         fill: #ccc;
# MAGIC         stroke: #fff;
# MAGIC       }
# MAGIC 
# MAGIC       path.arc {
# MAGIC         pointer-events: none;
# MAGIC         fill: none;
# MAGIC         stroke: #000;
# MAGIC         display: none;
# MAGIC       }
# MAGIC 
# MAGIC       path.cell {
# MAGIC         fill: none;
# MAGIC         pointer-events: all;
# MAGIC       }
# MAGIC 
# MAGIC       circle {
# MAGIC         fill: steelblue;
# MAGIC         fill-opacity: .8;
# MAGIC         stroke: #fff;
# MAGIC       }
# MAGIC 
# MAGIC       #cells.voronoi path.cell {
# MAGIC         stroke: brown;
# MAGIC       }
# MAGIC 
# MAGIC       #cells g:hover path.arc {
# MAGIC         display: inherit;
# MAGIC       }
# MAGIC     </style>
# MAGIC   </head>
# MAGIC   <body>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.csv.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geo.js"></script>
# MAGIC     <script src="https://mbostock.github.io/d3/talk/20111116/d3/d3.geom.js"></script>
# MAGIC     <script>
# MAGIC       var graph = $graph;
# MAGIC       var w = $width;
# MAGIC       var h = $height;
# MAGIC 
# MAGIC       var linksByOrigin = {};
# MAGIC       var countByAirport = {};
# MAGIC       var locationByAirport = {};
# MAGIC       var positions = [];
# MAGIC 
# MAGIC       var projection = d3.geo.azimuthal()
# MAGIC           .mode("equidistant")
# MAGIC           .origin([-98, 38])
# MAGIC           .scale(1400)
# MAGIC           .translate([640, 360]);
# MAGIC 
# MAGIC       var path = d3.geo.path()
# MAGIC           .projection(projection);
# MAGIC 
# MAGIC       var svg = d3.select("body")
# MAGIC           .insert("svg:svg", "h2")
# MAGIC           .attr("width", w)
# MAGIC           .attr("height", h);
# MAGIC 
# MAGIC       var states = svg.append("svg:g")
# MAGIC           .attr("id", "states");
# MAGIC 
# MAGIC       var circles = svg.append("svg:g")
# MAGIC           .attr("id", "circles");
# MAGIC 
# MAGIC       var cells = svg.append("svg:g")
# MAGIC           .attr("id", "cells");
# MAGIC 
# MAGIC       var arc = d3.geo.greatArc()
# MAGIC           .source(function(d) { return locationByAirport[d.source]; })
# MAGIC           .target(function(d) { return locationByAirport[d.target]; });
# MAGIC 
# MAGIC       d3.select("input[type=checkbox]").on("change", function() {
# MAGIC         cells.classed("voronoi", this.checked);
# MAGIC       });
# MAGIC 
# MAGIC       // Draw US map.
# MAGIC       d3.json("https://mbostock.github.io/d3/talk/20111116/us-states.json", function(collection) {
# MAGIC         states.selectAll("path")
# MAGIC           .data(collection.features)
# MAGIC           .enter().append("svg:path")
# MAGIC           .attr("d", path);
# MAGIC       });
# MAGIC 
# MAGIC       // Parse links
# MAGIC       graph.links.forEach(function(link) {
# MAGIC         var origin = graph.nodes[link.source].name;
# MAGIC         var destination = graph.nodes[link.target].name;
# MAGIC 
# MAGIC         var links = linksByOrigin[origin] || (linksByOrigin[origin] = []);
# MAGIC         links.push({ source: origin, target: destination });
# MAGIC 
# MAGIC         countByAirport[origin] = (countByAirport[origin] || 0) + 1;
# MAGIC         countByAirport[destination] = (countByAirport[destination] || 0) + 1;
# MAGIC       });
# MAGIC 
# MAGIC       d3.csv("https://mbostock.github.io/d3/talk/20111116/airports.csv", function(data) {
# MAGIC 
# MAGIC         // Build list of airports.
# MAGIC         var airports = graph.nodes.map(function(node) {
# MAGIC           return data.find(function(airport) {
# MAGIC             if (airport.iata === node.name) {
# MAGIC               var location = [+airport.longitude, +airport.latitude];
# MAGIC               locationByAirport[airport.iata] = location;
# MAGIC               positions.push(projection(location));
# MAGIC 
# MAGIC               return true;
# MAGIC             } else {
# MAGIC               return false;
# MAGIC             }
# MAGIC           });
# MAGIC         });
# MAGIC 
# MAGIC         // Compute the Voronoi diagram of airports' projected positions.
# MAGIC         var polygons = d3.geom.voronoi(positions);
# MAGIC 
# MAGIC         var g = cells.selectAll("g")
# MAGIC             .data(airports)
# MAGIC           .enter().append("svg:g");
# MAGIC 
# MAGIC         g.append("svg:path")
# MAGIC             .attr("class", "cell")
# MAGIC             .attr("d", function(d, i) { return "M" + polygons[i].join("L") + "Z"; })
# MAGIC             .on("mouseover", function(d, i) { d3.select("h2 span").text(d.name); });
# MAGIC 
# MAGIC         g.selectAll("path.arc")
# MAGIC             .data(function(d) { return linksByOrigin[d.iata] || []; })
# MAGIC           .enter().append("svg:path")
# MAGIC             .attr("class", "arc")
# MAGIC             .attr("d", function(d) { return path(arc(d)); });
# MAGIC 
# MAGIC         circles.selectAll("circle")
# MAGIC             .data(airports)
# MAGIC             .enter().append("svg:circle")
# MAGIC             .attr("cx", function(d, i) { return positions[i][0]; })
# MAGIC             .attr("cy", function(d, i) { return positions[i][1]; })
# MAGIC             .attr("r", function(d, i) { return Math.sqrt(countByAirport[d.iata]); })
# MAGIC             .sort(function(a, b) { return countByAirport[b.iata] - countByAirport[a.iata]; });
# MAGIC       });
# MAGIC     </script>
# MAGIC   </body>
# MAGIC </html>""")
# MAGIC   }
# MAGIC 
# MAGIC   def help() = {
# MAGIC displayHTML("""
# MAGIC <p>
# MAGIC Produces a force-directed graph given a collection of edges of the following form:</br>
# MAGIC <tt><font color="#a71d5d">case class</font> <font color="#795da3">Edge</font>(<font color="#ed6a43">src</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">dest</font>: <font color="#a71d5d">String</font>, <font color="#ed6a43">count</font>: <font color="#a71d5d">Long</font>)</tt>
# MAGIC </p>
# MAGIC <p>Usage:<br/>
# MAGIC <tt>%scala</tt></br>
# MAGIC <tt><font color="#a71d5d">import</font> <font color="#ed6a43">d3._</font></tt><br/>
# MAGIC <tt><font color="#795da3">graphs.force</font>(</br>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">height</font> = <font color="#795da3">500</font>,<br/>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">width</font> = <font color="#795da3">500</font>,<br/>
# MAGIC &nbsp;&nbsp;<font color="#ed6a43">clicks</font>: <font color="#795da3">Dataset</font>[<font color="#795da3">Edge</font>])</tt>
# MAGIC </p>""")
# MAGIC   }
# MAGIC }

# COMMAND ----------

tripDelays.registerTempTable("clicks")

# COMMAND ----------

display(tripDelays)

# COMMAND ----------

# MAGIC %md ##Showing number of flights which got delayed from philadelphia
# MAGIC Note:To see the delay route click on the blue node.
# MAGIC The size of the node is big, as many flights got delayed from philadelphia.

# COMMAND ----------

# MAGIC %scala
# MAGIC import d3a._
# MAGIC graphs.force(
# MAGIC   height = 800,
# MAGIC   width = 1200,
# MAGIC   clicks = sql("""select src, dst as dest, count(1) as count from clicks where src in ('PHL') and  delay>0 group by src, dst """).as[Edge])

# COMMAND ----------

# MAGIC %md ## Conclusion
# MAGIC 
# MAGIC This project and the analysis retrieved are useful not only for passengers point of view but for every decision-maker in the aviation industry.  Apart from the financial losses incurred by the industry, flight delay also portray a negative reputation of the airlines and decreases their reliability. This project can be used as a prototype by any aviation authority for their benefit; it can work as an efficient model to study delay analysis, based on the dataset.  This project has encompassed and showed the importance of  Various Analysis in Machine  Learning,  Data  Mining Concepts for efficient data cleaning,  Cross  Validation technique, and Regularization in ML for making proper models and its predictive analysis.

# COMMAND ----------

# MAGIC %md ##Recommendations
# MAGIC 
# MAGIC The results can be utilised by the airline companies to improve their functioning in order to perform better on the delays caused by Carrier Delay or Aircraft delay i.e. factors which could be controlled by the airlines in some way. For the other delay causes, which may not be directly controlled by airlines can be used to find alternative routes and services to improve customer satisfaction. Along with the airlines, the results of the analysis can also be utilised to recommend the best airlines that can be chosen by customers traveling on different routes and at different times. Customers can make use of the analysis and take more informed decisions based on the results.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
# MAGIC chmod +x gdown.pl

# COMMAND ----------

# MAGIC %sh
# MAGIC ./gdown.pl https://drive.google.com/open?id=1xAJiBigkSifAS2bsbfSsc-y44CMVDTMQ ML2_tableau.twb
# MAGIC ls

# COMMAND ----------

# MAGIC %md ### We downloaded the tableau files but unable to show here as it requires ODBC connection , so we have attached our findings as screen shot along with tableau original file in submission
