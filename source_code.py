# BIG DATA PROJECT
# NAME - MATRICULATION NUMBER
# Abhishek Harsha Raj- 219203362
# Anirudh Kudaragundi Anand Rao- 219203189
# Harish Muralidhar Rao- 219203058
# Sindhoora Hegde- 219203260

# The Dataset considered is of football players with many attribute columns from FIFA 2019. 
# The Preprocessing involves removing columns which are not needed for analysis, removing unwanted symbols, removing null values and feature scaling
# Analysis and Visualization includes the data and plots which can help a football manager to take decisions. They are as follows
# 1. Finding the attributes or columns with highest correlation and plotting those attributes
# 2. Determining the change in market value and overall rating of players with age
# 3. Analyzing top 10 best players in each position, their average market value and average overall rating and Visualizing them appropriately
# 4. Analyzing 10 least worthy players in each position, their average market value and average overall rating and Visualizing them appropriately
# 5. KMeans clustering is used to cluster based on highest correlated columns computed above with respect to position and subplots for each cluster is plotted. Also Cluster centers and prediction columns are displayed


# Importing required libraries for the analysis
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import when
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.stat import Correlation
from pyspark.sql import Window
from pyspark.sql.functions import rank, col
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import ScalarFormatter
import sys

# Defining encoding for stdout as utf-8 in order to display foreign names with special characters without any error 
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

# Creating spark session
spark = SparkSession.builder.master("local[*]").appName("BigData_Project").getOrCreate()

# Reading csv file into a spark sql dataframe
dataframe = spark.read.format("csv").option("inferSchema","true").option("encoding", "utf8").option("header","true").load("data.csv")

# DATA PREPROCESSING

# Removing the columns which are not necessary for our set of analysis
dataframe=dataframe.drop('_c0','Photo','Flag','Club Logo','Wage','Special'
,'Preferred Foot','International Reputation','Weak Foot','Skill Moves','Work Rate','Body Type','Real Face','Jersey Number'
,'Joined','Loaned From','Contract Valid Until','Height','Weight','LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM'
,'LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys'
,'Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina'
,'Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle','SlidingTackle','GKDiving'
,'GKHandling','GKKicking','GKPositioning','GKReflexes','Release Clause')

# Removing unwanted symbols and converting Value attribute into floating values needed for analysis
dataframe = dataframe.withColumn('Value', regexp_replace('Value','€', ''))
millions=dataframe.filter(dataframe["Value"].endswith('M'))
millions = millions.withColumn('Value', regexp_replace('Value','M',''))
millions = millions.withColumn('Value', when(millions.Value.isNull(),0).otherwise(millions["Value"].cast("float")*1000000))
thousands=dataframe.filter(dataframe["Value"].endswith('K'))
thousands = thousands.withColumn('Value', regexp_replace('Value','K',''))
thousands = thousands.withColumn('Value', when(thousands.Value.isNull(),0).otherwise(thousands["Value"].cast("float")*1000))
cleaned_set=millions.union(thousands)

# User defined function for feature scaling using min-max normalization
unlist = udf(lambda x: round(float(list(x)[0]),4), DoubleType())
for i in ["Age","Overall","Potential","Value"]:
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")
    pipeline = Pipeline(stages=[assembler, scaler])
    cleaned_set = pipeline.fit(cleaned_set).transform(cleaned_set).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i+"_Vect")

# Droping rows which contains null values
cleaned_set=cleaned_set.na.drop()

# Displaying the cleaned data set sample 5 rows out of many
print("\nThe cleaned dataset with sample of 5 rows\n")
cleaned_set.show(5)

# DATA ANALYSIS

# Finding correlation between Overall player ratings, Age and Potential of players with market value of each player
correlation_Overall_Value=cleaned_set.corr("Overall_Scaled","Value_Scaled")
correlation_Age_Value=cleaned_set.corr("Age_Scaled","Value_Scaled")
correlation_Potential_Value=cleaned_set.corr("Potential_Scaled","Value_Scaled")
correlation_list=[correlation_Overall_Value,correlation_Age_Value,correlation_Potential_Value]

# Finding the two columns which have maximum correlation and print its value
max_correlation=max(correlation_list)
# Displaying maximum Pearson correlation between Overall player ratings, Age and Potential of players with market value of each player
print("\nThe value of Maximum correlation computed is\n")
print(max_correlation)
print("The max correlation is observed between Overall player ratings and market value")


# The maximum correlation observed is for Overall player ratings and market value of each player
# Plotting Overall player ratings and market value of each player along with required plot properties
g=cleaned_set.toPandas()
graph1 = g.plot(x='Overall',y='Value',kind='scatter',color='green', figsize=(25,50),label='players', legend=True, fontsize=20)
graph1.set_title("Overall Player ratings vs Market Value of players",fontsize=30)
graph1.set_xlabel("Overall Player ratings", fontsize=20)
graph1.set_ylabel("Player Market Value in Millions", fontsize=20)
graph1.set_xlim(40,100)
graph1.set_ylim(0,120000000)
graph1.xaxis.set_major_locator(tck.MultipleLocator(5))
graph1.yaxis.set_major_locator(tck.MultipleLocator(30000000))
plt.show()

# Visualizing relation between age with market value and overall player rating

# Plotting how market value of players vary with age with required plot properties
# The mean market value for each age group is considered while plotting
agevsavgvalue_dataset=cleaned_set.groupBy('Age').agg({'Value': 'avg'})
g=agevsavgvalue_dataset.toPandas()
graph2 = g.plot(x='Age',y='avg(Value)',kind='scatter',color='blue', figsize=(25,50),label='players', legend=True, fontsize=20)
graph2.set_title("Market Value vs Age of players in FIFA 19",fontsize=30)
graph2.set_xlabel("Players Age", fontsize=20)
graph2.set_ylabel("Player Market Value in Millions", fontsize=20)
plt.show()

# Plotting how overall player rating varies with age with required plot properties
# The mean overall rating for each age group is considered while plotting
agevsavgoverall_dataset=cleaned_set.groupBy('Age').agg({'Overall': 'avg'})
g=agevsavgoverall_dataset.toPandas()
graph2 = g.plot(x='Age',y='avg(Overall)',kind='scatter',color='blue', figsize=(25,50),label='players', legend=True, fontsize=20)
graph2.set_title("Average Overall Player ratings vs Age of players in FIFA 19",fontsize=30)
graph2.set_xlabel("Players Age", fontsize=20)
graph2.set_ylabel("Player Overall ratings", fontsize=20)
graph2.set_xlim(15,45)
graph2.set_ylim(50,100)
graph2.xaxis.set_major_locator(tck.MultipleLocator(5))
graph2.yaxis.set_major_locator(tck.MultipleLocator(5))
plt.show()

# Finding top 10 best players with respect to each position considering their overall
# Adding a rank column to rank the best players in each position
window_for_highest_overall = Window.partitionBy(cleaned_set['Position']).orderBy(cleaned_set['Overall'].desc())
top_worthy_players=cleaned_set.select('*', rank().over(window_for_highest_overall).alias('rank')).filter(col('rank')<=10)
print("\nThe top 10 best players who are worthy based on their Overall for each position are\n")
print("The column rank indicates the rank of players")
top_worthy_players.show(1000)

# Plotting average market value of top 10 worthy players for each position with required plot properties
# Adding avg(Value) column containing average of market values for each position
positionvsavgvalue_dataset=top_worthy_players.groupBy('Position').agg({'Value': 'avg'})
g=positionvsavgvalue_dataset.toPandas()
graph3 = g.plot(x='Position',y='avg(Value)',kind='bar',color='green', figsize=(25,50),label='players', legend=True, fontsize=20)
graph3.set_title("Position vs Average Market Value of best players",fontsize=20)
graph3.set_xlabel("Position", fontsize=10)
graph3.set_ylabel("Player Market Value in Millions", fontsize=10)
plt.show()

# Plotting average overall of top 10 worthy players for each position with required plot properties
# Adding avg(Overall) column containing average of overall rating for each position
positionvsavgoverall_dataset=top_worthy_players.groupBy('Position').agg({'Overall': 'avg'})
g=positionvsavgoverall_dataset.toPandas()
graph3 = g.plot(x='Position',y='avg(Overall)',kind='bar',color='green', figsize=(25,50),label='players', legend=True, fontsize=20)
graph3.set_title("Position vs Average Overall player ratings of best players",fontsize=20)
graph3.set_xlabel("Position", fontsize=10)
graph3.set_ylabel("Overall Player ratings", fontsize=10)
plt.show()

# Finding 10 least worthy players with respect to each position considering their overall
# Adding a rank column to rank the least worthy players in each position
window_for_least_overall = Window.partitionBy(cleaned_set['Position']).orderBy(cleaned_set['Overall'].asc())
least_worthy_players=cleaned_set.select("*",rank().over(window_for_least_overall).alias('rank')).filter(col('rank')<=10)
print("\nThe 10 least worthy players who are less worthy based on their Overall for each position are\n")
print("The column rank indicates the rank of players")
least_worthy_players.show(1000)

# Plotting average market value of 10 least worthy players for each position with required plot properties
positionvsavgvalue_dataset=least_worthy_players.groupBy('Position').agg({'Value': 'avg'})
g=positionvsavgvalue_dataset.toPandas()
graph3 = g.plot(x='Position',y='avg(Value)',kind='bar',color='red', figsize=(25,50),label='players', legend=True, fontsize=20)
graph3.set_title("Position vs Average Market Value of least worthy players ",fontsize=30)
graph3.set_xlabel("Position", fontsize=20)
graph3.set_ylabel("Player Value in Millions", fontsize=20)
plt.show()

# Plotting average overall of 10 least worthy players for each position with required plot properties
positionvsavgoverall_dataset=least_worthy_players.groupBy('Position').agg({'Overall': 'avg'})
g=positionvsavgoverall_dataset.toPandas()
graph3 = g.plot(x='Position',y='avg(Overall)',kind='bar',color='red', figsize=(25,50),label='players', legend=True, fontsize=20)
graph3.set_title("Position vs Average Overall player ratings of least worthy players",fontsize=30)
graph3.set_xlabel("Position", fontsize=20)
graph3.set_ylabel("Overall ratings", fontsize=20)
plt.show()


# Using k-means clustering between overall player ratings and market value as these two attributes have the highest correlation for the best players
# Building a K-means model 
vecAssembler = VectorAssembler(inputCols=["Overall_Scaled", "Value_Scaled"], outputCol="features")
new_df = vecAssembler.transform(top_worthy_players)
kmeans = KMeans(k=9, seed=1)  
model = kmeans.fit(new_df.select('features'))
transformed = model.transform(new_df)

# Finding cluster centers and displaying them, calculating count of number of clusters
wssse = model.computeCost(new_df)
print("\nWithin Set Sum of Squared Errors = " + str(wssse))
count=0
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print("Cluster K:",count)
    print("center:",center)
    count=count+1

# Displaying clusters as a column named Prediction for the best players as computed earlier
print("\n\nThe predicted cluster values after performing KMeans Clustering\n")
print("The column name prediction shows cluster k for each position")
transformed.show(500)    

# Plotting clusters Based on Overall of Players and Market Value as they represent highest correlation grouped with position, with required plot properties annotated with positions
temp=transformed.select("Overall","Value","prediction","Position")
g=temp.toPandas()

# subplots with clusters 1,2,3
# Plot All Clusters
plt.subplot(221)
plt.scatter(g["Overall"], g["Value"], c=g["prediction"])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("All clusters based on Overall player rating and Market Value")
plt.legend(["Players"])

# Plot cluster 1
plt.subplot(222)
g1=g.loc[g['prediction']==1]
plt.scatter(g1['Overall'],g1['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 1 values")
plt.legend(["Players"])
x = g1['Overall'].tolist()
y= g1['Value'].tolist()
n= g1['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 2
plt.subplot(223)
g2=g.loc[g['prediction']==2]
plt.scatter(g2['Overall'],g2['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 2 values")
plt.legend(["Players"])
x = g2['Overall'].tolist()
y= g2['Value'].tolist()
n= g2['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 3
plt.subplot(224)
g3=g.loc[g['prediction']==3]
plt.scatter(g3['Overall'],g3['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 3 values")
plt.legend(["Players"])
x = g3['Overall'].tolist()
y= g3['Value'].tolist()
n= g3['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

plt.tight_layout()
plt.show()

# subplots with clusters 4,5,6
# plot All Clusters
plt.subplot(221)
plt.scatter(g["Overall"], g["Value"], c=g["prediction"])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("All clusters based on overall player rating and market value")
plt.legend(["Players"])

# plot cluster 4
plt.subplot(222)
g4=g.loc[g['prediction']==4]
plt.scatter(g4['Overall'],g4['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 4 values")
plt.legend(["Players"])
x = g4['Overall'].tolist()
y= g4['Value'].tolist()
n= g4['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 5
plt.subplot(223)
g5=g.loc[g['prediction']==5]
plt.scatter(g5['Overall'],g5['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 5 values")
plt.legend(["Players"])
x = g5['Overall'].tolist()
y= g5['Value'].tolist()
n= g5['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 6
plt.subplot(224)
g6=g.loc[g['prediction']==6]
plt.scatter(g6['Overall'],g6['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 6 values")
plt.legend(["Players"])
x = g6['Overall'].tolist()
y= g6['Value'].tolist()
n= g6['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

plt.tight_layout()
plt.show()

# subplots with clusters 7,8,0
# plot All clusters
plt.subplot(221)
plt.scatter(g["Overall"], g["Value"], c=g["prediction"])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("All Clusters based on overall player rating and market value")
plt.legend(["Players"])

# plot cluster 7
plt.subplot(222)
g7=g.loc[g['prediction']==7]
plt.scatter(g7['Overall'],g7['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 7 values")
plt.legend(["Players"])
x = g7['Overall'].tolist()
y= g7['Value'].tolist()
n= g7['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 8
plt.subplot(223)
g8=g.loc[g['prediction']==8]
plt.scatter(g8['Overall'],g8['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 8 values")
plt.legend(["Players"])
x = g8['Overall'].tolist()
y= g8['Value'].tolist()
n= g8['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

# plot cluster 0
plt.subplot(224)
g9=g.loc[g['prediction']==0]
plt.scatter(g9['Overall'],g9['Value'])
plt.xlabel("Overall Player Rating")
plt.ylabel("Player Market Value")
plt.title("Cluster 0 values")
plt.legend(["Players"])
x = g9['Overall'].tolist()
y= g9['Value'].tolist()
n= g9['Position'].tolist()
for i,txt in enumerate(n):
    text=plt.annotate(txt,(x[i],y[i]))
    text.set_fontsize(10)

plt.tight_layout()
plt.show()

