# ## K-means Clustering

# In[39]:
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Clustering using K-Means').getOrCreate()

data_stock=spark.read.csv('Nasdaq_1.csv', header=True, inferSchema=True)

data_stock.printSchema()


# In[40]:
# checking NA value 
data_stock=data_stock.na.drop()


# In[44]:
# Vectorize column 
from pyspark.ml.feature import VectorAssembler
data_stock.columns

assemble=VectorAssembler(inputCols=[
    'Open',
    'High',
    'Low',
    'Close',
    'Volume'], outputCol='features')

assembled_data=assemble.transform(data_stock)

assembled_data.show(2)


# In[46]:
from pyspark.ml.feature import StandardScaler

scale=StandardScaler(inputCol='features', outputCol='standardized')

data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)

data_scale_output.show(2)


# In[47]:
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_score=[]
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',                                 metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2,10):
    
    KMeans_algo=KMeans(featuresCol='standardized', k=i)
    KMeans_fit=KMeans_algo.fit(data_scale_output)
    output=KMeans_fit.transform(data_scale_output)
       
    
    score=evaluator.evaluate(output)
    
    silhouette_score.append(score)
    
    print("Silhouette Score:",score)

#Visualizing the silhouette scores in a plot

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('cost')
### create a blank dataframe
close=pd.DataFrame()

df2_GOOGL = df2[df2['Symbol'] == 'GOOGL']
close['google'] = df2_GOOGL['Close'].reset_index(drop=True)

df2_AAPL = df2[df2['Symbol'] == 'AAPL']
close['apple'] = df2_AAPL['Close'].reset_index(drop=True)

df2_AMZN = df2[df2['Symbol'] == 'AMZN']
close['amazon'] = df2_AMZN['Close'].reset_index(drop=True)

df2_MSFT = df2[df2['Symbol'] == 'MSFT']
close['microsoft'] = df2_MSFT['Close'].reset_index(drop=True)

df2_META = df2[df2['Symbol'] == 'META']
close['meta'] = df2_META['Close'].reset_index(drop=True)

df2_ADBE = df2[df2['Symbol'] == 'ADBE']
close['adobe'] = df2_ADBE['Close'].reset_index(drop=True)

df2_AMAT = df2[df2['Symbol'] == 'AMAT']
close['applied_materials'] = df2_AMAT['Close'].reset_index(drop=True)

df2_AMD = df2[df2['Symbol'] == 'AMD']
close['advanced_micro_devices'] = df2_AMD['Close'].reset_index(drop=True)

df2_AMGN = df2[df2['Symbol'] == 'AMGN']
close['amgen'] = df2_AMGN['Close'].reset_index(drop=True)

df2_AVGO = df2[df2['Symbol'] == 'AVGO']
close['broadcom'] = df2_AVGO['Close'].reset_index(drop=True)

df2_CMCSA = df2[df2['Symbol'] == 'CMCSA']
close['comcast'] = df2_CMCSA['Close'].reset_index(drop=True)

df2_COST = df2[df2['Symbol'] == 'COST']
close['costco'] = df2_COST['Close'].reset_index(drop=True)

df2_CSCO = df2[df2['Symbol'] == 'CSCO']
close['cisco_systems'] = df2_CSCO['Close'].reset_index(drop=True)

df2_HON = df2[df2['Symbol'] == 'HON']
close['honeywell'] = df2_HON['Close'].reset_index(drop=True)

df2_INTC = df2[df2['Symbol'] == 'INTC']
close['intel'] = df2_INTC['Close'].reset_index(drop=True)

df2_INTU = df2[df2['Symbol'] == 'INTU']
close['intuit'] = df2_INTU['Close'].reset_index(drop=True)

df2_NFLX = df2[df2['Symbol'] == 'NFLX']
close['netflix'] = df2_NFLX['Close'].reset_index(drop=True)

df2_NVDA = df2[df2['Symbol'] == 'NVDA']
close['nvidia'] = df2_NVDA['Close'].reset_index(drop=True)

df2_PEP = df2[df2['Symbol'] == 'PEP']
close['pepsico'] = df2_PEP['Close'].reset_index(drop=True)

df2_QCOM = df2[df2['Symbol'] == 'QCOM']
close['qualcomm'] = df2_QCOM['Close'].reset_index(drop=True)

df2_SBUX = df2[df2['Symbol'] == 'SBUX']
close['starbucks'] = df2_SBUX['Close'].reset_index(drop=True)

df2_TMUS = df2[df2['Symbol'] == 'TMUS']
close['t-mobile'] = df2_TMUS['Close'].reset_index(drop=True)

df2_TSLA = df2[df2['Symbol'] == 'TSLA']
close['tesla'] = df2_TSLA['Close'].reset_index(drop=True)

df2_TXN = df2[df2['Symbol'] == 'TXN']
close['texas_instruments'] = df2_TXN['Close'].reset_index(drop=True)

print(close)

close.head()

import seaborn as sns

sns.heatmap(close.corr(),annot=True,cmap='gray_r',linecolor="black")

plt.figure(figsize=(15, 12))

sns.heatmap(close.corr(), annot=True, cmap='gray_r', linecolor="black")

plt.show()

companies = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
filtered_data = df[df['Symbol'].isin(companies)]

plt.figure(figsize=(15, 10))
for i, company in enumerate(companies, 1):
    plt.subplot(2, 2, i)
    df_temp = df[df['Symbol'] == company]
    plt.plot(df_temp['Date'], df_temp['Volume'])
    plt.title(company)

plt.tight_layout()
plt.show()

print(close)

print(close.isnull().sum())

close=close.dropna()

print(close)

close_reversed = close.reset_index(drop=True)

# Reverse the index of the DataFrame
close = close[::-1]

print(df_reversed)

print(close)

#Calculate the annual mean returns and variances
daily_returns = close.pct_change()
annual_mean_returns = daily_returns.mean() * 252
annual_return_variance = daily_returns.var() * 252

# Create a new dataframe
close2 = pd.DataFrame(close.columns, columns=['Company_name'])
close2['Variances'] = annual_return_variance.values
close2['Returns'] = annual_mean_returns.values

# Show the data
close2

from sklearn.cluster import KMeans

# Use the Elbow method to determine the number of clusters to use to group the stocks
# Get and store the annual returns and annual variances
X = close2[['Returns', 'Variances']].values
inertia_list=[]
for k in range(2,16):
  #Create and train the model
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(X)
  inertia_list.append(kmeans.inertia_)

# Plot the data
plt.plot(range(2,16), inertia_list)
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia or Sum Squared Error (SSE)')
plt.show()

# Get an show the lables / groups
kmeans = KMeans(n_clusters=5).fit(X)
labels = kmeans.labels_
labels

close2['Cluster_Labels'] = labels
close2

# Plot and show the different clusters
plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
plt.title('K-Means Plot')
plt.xlabel('Returns')
plt.ylabel('Variances')
plt.show()

# Create a function to build a simple divesed portfolio
def diversed_port():
  for i in range(0, 5):
    symbol = close2[ close2['Cluster_Labels'] == i].head(1)
    print(symbol[['Company_name', 'Cluster_Labels']])

diversed_port()

# Create a new dataframe
close3 = pd.DataFrame(close.columns, columns=['Company_name'])
close3['Variances'] = annual_return_variance.values
close3['Returns'] = annual_mean_returns.values

# Get an show the lables / groups in number 4
kmeans2 = KMeans(n_clusters=4)
labels2 = kmeans2.fit_predict(X)

# show the data

close3

close3['Cluster_Labels'] = labels2
close3

# Plot and show the different clusters
plt.scatter(X[:,0], X[:,1], c = labels2, cmap = 'rainbow')
plt.title('K-Means Plot')
plt.xlabel('Returns')
plt.ylabel('Variances')
plt.show()

close.columns

# Create a new dataframe
close4 = pd.DataFrame(close.columns, columns=['Company_name'])
close4['Variances'] = annual_return_variance.values
close4['Returns'] = annual_mean_returns.values

# Get an show the lables / groups in number 3
kmeans3 = KMeans(n_clusters=3)
labels3 = kmeans3.fit_predict(X)

close4
close4['Cluster_Labels'] = labels3
close4

# Plot and show the different clusters
plt.scatter(X[:,0], X[:,1], c = labels3, cmap = 'rainbow')
plt.title('K-Means Plot')
plt.xlabel('Returns')
plt.ylabel('Variances')
plt.show()

# Create a function to build a simple divesed portfolio
def diversed_port():
  for i in range(0, 3):
    symbol = close4[ close4['Cluster_Labels'] == i].head(1)
    print(symbol[['Company_name', 'Cluster_Labels']])

diversed_port()

# Group the data by cluster labels
cluster_groups = close4.groupby('Cluster_Labels')

# Calculate the average return and variance for each cluster
average_returns = cluster_groups['Returns'].mean()
average_variances = cluster_groups['Variances'].mean()

# Display the average returns and variances for each cluster
cluster_statistics = pd.DataFrame({
    'Cluster_Labels': average_returns.index,
    'Average_Returns': average_returns.values,
    'Average_Variances': average_variances.values
})

cluster_statistics

# Calculate the Sharpe Ratio
cluster_statistics['Sharpe_Ratio'] = cluster_statistics['Average_Returns'] / np.sqrt(cluster_statistics['Average_Variances'])

# Display the updated DataFrame
print(cluster_statistics)

# Calculate the Sharpe Ratio
close4['Sharpe_Ratio'] = close4['Returns'] / np.sqrt(close4['Variances'])

# Display the updated DataFrame
print(close4)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Box plot for Sharpe Ratio grouped by Cluster_Labels
sns.boxplot(data=close4, x='Cluster_Labels', y='Sharpe_Ratio', ax=ax)

# Set labels and title
ax.set_xlabel('Cluster')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio Distribution by Cluster')

# Display the plot
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a new dataframe with columns for cluster labels, returns, and variances
data = pd.DataFrame()
data['Cluster_Labels'] = close4['Cluster_Labels']
data['Returns'] = close4['Returns']
data['Variances'] = close4['Variances']

# Set the color palettes for returns and variances
return_color_palette = ['red', 'green', 'blue']
variance_color_palette = ['orange', 'purple', 'cyan']

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Box plot for Returns
sns.boxplot(data=data, x='Cluster_Labels', y='Returns', ax=ax, width=0.4, color=return_color_palette[0],
            boxprops=dict(edgecolor='black'),
            capprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            medianprops=dict(color='black'),
            flierprops=dict(markeredgecolor='black'))

# Box plot for Variances of Return
sns.boxplot(data=data, x='Cluster_Labels', y='Variances', ax=ax, width=0.4, color=variance_color_palette[0],
            boxprops=dict(edgecolor='black'),
            capprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            medianprops=dict(color='black'),
            flierprops=dict(markeredgecolor='black'))

# Set labels and title
ax.set_xlabel('Cluster')
ax.set_ylabel('Value')
ax.set_title('Distribution of Returns and Variances of Return by Cluster')

# Create custom legend handles
return_patch = plt.Line2D([], [], marker='s', color='white', markerfacecolor='white', markersize=10, markeredgecolor=return_color_palette[0])
variance_patch = plt.Line2D([], [], marker='s', color='white', markerfacecolor='white', markersize=10, markeredgecolor=variance_color_palette[0])

# Create the legend
legend = ax.legend([return_patch, variance_patch], ['Returns', 'Variances of Return'], loc='upper left', bbox_to_anchor=(0.05, 0.95))

# Display the plot
plt.show()

# Create a new dataframe
close5 = pd.DataFrame(close.columns, columns=['Company_name'])
close5['Variances'] = annual_return_variance.values
close5['Returns'] = annual_mean_returns.values

# Get an show the lables / groups in number 6
kmeans4 = KMeans(n_clusters=6)
labels4 = kmeans4.fit_predict(X)

close5['Cluster_Labels'] = labels4
close5

# Plot and show the different clusters
plt.scatter(X[:,0], X[:,1], c = labels4, cmap = 'rainbow')
plt.title('K-Means Plot')
plt.xlabel('Returns')
plt.ylabel('Variances')
plt.show()
