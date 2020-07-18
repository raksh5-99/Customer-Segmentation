<h1>Customer Segmentation Using Python Machine Learning</h1>

<h2>Customer Segmentation</h2>
   
   <h4>Whenever you need to find your best customer, customer segmentation is the ideal methodology.Customer Segmentation is one the most important applications of unsupervised learning. Using clustering techniques, companies can identify the several segments of customers allowing them to target the potential user base. In this machine learning project, we will make use of K-means clustering which is the essential algorithm for clustering unlabeled dataset.</h4>
    
<h2>Defintion:</h2>
    <h4>Customer Segmentation is the process of division of customer base into several groups of individuals that share a similarity in different ways that are relevant to marketing such as gender, age, interests, and miscellaneous spending habits.</h4>
    
![imagename](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/07/R-project-customer-segmentation.png)
    
<h2>Procedure involved in Customer Segmentation Technique</h2>
<h4>Determine the need of the Segment.</h4>
<h4>You have to think of this in terms of consumption by customers or what would each of your customer like to have.For example – In a region, there are many normal restaurants but there is no Italian restaurant or there is no fast food chain. So, you came to know the NEED of consumers in that specific region.</h4>

![image1](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-1.jpg)

<h4>Identifying the Segment</h4>
<h4>Once you know the need of the customers, you need to identify that “who” will be the customers to choose your product over other offerings. Quite simply, you have to decide which type of segmentation you are going to use in this case. Is it going to be geographic, demographic, psychographic or what? The 1st step gives you a mass of crowd, and in the 2nd step, you have to differentiate the people from within that crowd.
  
![image2](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-2.jpg)

<h4>Which Segment is most Attractive?</h4>
<h4>Out of the various segments you have identified via demography, geography or psychography, you have to choose which is the most attractive segment for you.Attractiveness of the firm also depends on the competition available in the segment.
Taking the above example of an Italian restaurant, the restaurant owner realizes that he has more middle aged people and youngsters in his vicinity.So the 1st target is the middle aged group, and the 2nd target is youngsters.</h4>

![image3](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-3.jpg)

<h4>Is the Segment giving Profit?</h4>
<h4>Example – The Italian restaurant owner above decides that he is getting fantastic profitability from the middle aged group, but he is getting poor profitability from youngsters. Youngsters like fast food and they like socializing. So they order very less, and spend a lot of time at the table, thereby reducing the profitability. So what does the owner do? How does he change this mindset when one of the segments he has identified is less profitable? Lets find out in the 5th step.</h4>

![image4](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-4.jpg)

<h4>Positioning for the Segment</h4>
<h4>If the firm wants a customer to buy their product, what is the value being provided to the customer, and in his mindset, where does the customer place the brand after purchasing the product? What was the value of the product to the customer and how valuable does he think the brand is – that is the work of positioning. And to complete the process of segmentation, you need to position your product in the mind of your segments.</h4>
<h4>Example – In the above case we saw that the Italian restaurant owner was finding youngsters unprofitable.He starts a fast food chain right next to the Italian restaurant. What happens is, although the area has other fast food restaurants, his restaurant is the only one which offers good Italian cuisine and a good fast food restaurant next door itself. So both, the middle aged target group and the youngsters can enjoy.</h4>

![image5](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-5.jpg)

<h4>Expanding the Segment</h4
<h4>All segments need to be scalable. So, if you have found a segment, that segment should be such that the business is able to expand with the type of segmentation chosen.In the above example, the Italian restaurant owner has the best process in his hand – an Italian restaurant combined with a fast food chain. He was using both Demographic and geographic segmentation. Now he starts looking at other geographic segments in other regions where he can establish the same concept and expand his business.</h4>

![image6](https://www.marketing91.com/wp-content/uploads/2016/05/Steps-in-Market-segmentation-6.jpg)

<h4>Incorporating the Segmentation to Maketing Strategy</h4>
<h4>Once you have found a segment which is profitable and expandable, you need to incorporate that segment in your marketing strategy.With the steps of market segmentation, your segments become clear and then you can adapt other variables of marketing strategy as per the segment being targeted. You can modify the products, keep the optimum price, enhance the distribution and the place and finally promote clearly and crisply to your target audience. Business becomes simpler due to the process of market segmentation.</h4>

<h2>Problem Statement</h2>
<h4>You are the owner of a mall and have basic information about the customers such as gender,age,annual income and spending score.Using customer segmentation technique decide your target customers.</h4>
<h2>Data</h2>
<h4>The dataset needed is [here](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)</h4>

<h2>Code</h2>

<h4>Import the required libraries</h4>

    ```
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as ply
    import seaborn as sns
    %matplotlib inline
    ```
<h4>Read Mall_Customers csv file as Dataframe called customer.</h4>

    ```
    cust=pd.read_csv("Mall_Customers.csv")
    cust.head()
    ```
<h4>Number of columns:</h4>

    ```
    customer.info()
    ```
<h4>To compute the summary of stastics pertaining to DataFrame columns</h4>
    
    ```
    customer.describe()
    ```
<h4>Constructing histogram based on age frequency</h4>

    ```
    sns.set_style('whitegrid')
    customer['Age'].hist(bins=30)
    plt.xlabel('Age')
    ```
<h4>Construction of jointplot() allows you to match up two distplots for bivariate data.Here we consider Age and Annual Income.</h4>

    ```
    sns.jointplot(x='Age',y='Annual Income (k$)',data=customer)
    ```
<h4>Constructing jointplot() of kind "kde" by considering Age and Spending Score.</h4>

    ```
    sns.jointplot(x='Age',y='Spending Score (1-100)',data=customer,color='blue',kind='kde')
    ```
<h4>A boxplot for Annual Income and Spending score for better understanding of distribution range. Here we clealy come to know that distribution range of Spending score is more than Annual Income</h4>

    ```
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    sns.boxplot(y=customer["Spending Score (1-100)"],color="green")
    plt.subplot(1,2,2)
    sns.boxplot(y=customer["Annual Income (k$)"],color="yellow")
    plt.show()
    ```
<h4>pairplot() which will plot pairwise relationships across an entire dataframe.</h4>

    ```
    customer.drop(["CustomerID"], axis = 1, inplace=True)
    sns.pairplot(customer,hue='Gender',palette='Set1')
    ```
<h4>A Bar plot to check the distribution of Male and Female in the dataset. Its shows that female population is more than the male population</h4>

    ```
    gender=customer.Gender.value_counts()
    sns.barplot(x=gender.index,y=gender.values)
    ```
<h4>Plotting the Barplot to know the distribution of Customers in each age-group.</h4>

    ```
    age0=customer.Age[(customer.Age<20)]
    age1= customer.Age[(customer.Age>=20)&(customer.Age<=30)]
    age2=customer.Age[(customer.Age>30)&(customer.Age<=40)]
    age3=customer.Age[(customer.Age>40)&(customer.Age<=50)]
    age4=customer.Age[(customer.Age>50)&(customer.Age<=60)]
    age5=customer.Age[(customer.Age>60)]
    x=['Below 20','21-30','31-40','41-50','51-60','60+']
    y=[len(age0.values),len(age1.values),len(age2.values),len(age3.values),len(age4.values),len(age5.values)]
    sns.barplot(x=x,y=y,palette='Accent')
    plt.title(str("Number of customers based on Age Group"))
    plt.xlabel(str("Age"))
    plt.ylabel(str("Number of customers"))
    plt.show()
    ```
<h4>A bar plot to visualize the number of customers according to their annual income.</h4>

    ```
    plt.figure(figsize=(15,7))
    income0=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]<15)]
    income1=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>=15)&(customer["Annual Income (k$)"]<=30)]
    income2=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>30)&(customer["Annual Income (k$)"]<=45)]
    income3=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>45)&(customer["Annual Income (k$)"]<=60)]
    income4=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>60)&(customer["Annual Income (k$)"]<=75)]
    income5=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>75)&(customer["Annual Income (k$)"]<=90)]
    income6=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>90)&(customer["Annual Income (k$)"]<=105)]
    income7=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>105)&(customer["Annual Income (k$)"]<=120)]
    income8=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>120)&(customer["Annual Income (k$)"]<=135)]
    income9=customer["Annual Income (k$)"][(customer["Annual Income (k$)"]>135)&(customer["Annual Income (k$)"]<=150)]
    x=['Below 15','15-30','31-45','46-60','61-75','76-90','91-105','106-120','121-135','136-150']
    y=    [len(income0.values),len(income1.values),len(income2.values),len(income3.values),len(income4.values),len(income5.values),len(income6.values),len(income7.values),len(income8.v      alues),len(income9.values)]
    sns.barplot(x=x,y=y)
    plt.title("Number of Customers based on Annual Income")
    plt.xlabel("Annual income(k$)")
    plt.ylabel("Number of customers")
    ```
<h2>K-means Clustering</h2>
<h4>It is a clustering algorithm that aims to partition n observations into k clusters.</h4>
<h4>Summing up the K-means clustering –</h4>

<h4>~We specify the number of clusters that we need to create.</h4>
<h4>~The algorithm selects k objects at random from the dataset. This object is the initial cluster or mean.</h4>
<h4>~The closest centroid obtains the assignment of a new observation. We base this assignment on the Euclidean Distance between object and the centroid.</h4>
<h4>~k clusters in the data points update the centroid through calculation of the new mean values present in all the data points of the cluster. The kth cluster’s centroid has a length of p that contains means of all variables for observations in the k-th cluster. We denote the number of variables with p.</h4>
<h4>~Iterative minimization of the total within the sum of squares. Then through the iterative minimization of the total sum of the square, the assignment stop wavering when we achieve maximum iteration. The default value is 10 that the R software uses for the maximum iterations.</h4>

<h4>Within Cluster Sum Of Squares (WCSS)</h4>
<h4>WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids.</h4>

<h4>we have to plot Within Cluster Sum Of Squares (WCSS) against the the number of clusters (K Value) to figure out the optimal number of clusters value.</h4>

    ```
    wcss=[]
    gender=customer.Gender.value_counts()
    for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(customer.iloc[:,2:])
    wcss.append(kmeans.inertia_)
    ```
    ```
    plt.grid()
    plt.plot(range(1,11),wcss, linewidth=1, color="green", marker ="8")
    plt.xlabel("K Value")
    plt.title("Within-Cluster-of-Squares ")
    plt.xticks(np.arange(1,11,1))
    plt.ylabel("WCSS")
    plt.show()
    ```
    ```
    kmeans.labels_
    ```


<h2>Determining Optimal Clusters using Elbow Method</h2>

<h2>minimize(sum W(Ck)), k=1…k</h2>

<h4>The main goal behind cluster partitioning methods like k-means is to define the clusters such that the intra-cluster variation stays minimum.</h4>
<h4>First, we calculate the clustering algorithm for several values of k. This can be done by creating a variation within k from 1 to 10 clusters.</h4>
<h4>We then calculate the total intra-cluster sum of square.</h4>
<h4>Then, we proceed to plot based on the number of k clusters. This plot denotes the appropriate number of clusters required in our model. In the plot, the location of a bend or a knee is the indication of the optimum number of clusters.</h4>

    ```
    km = KMeans(n_clusters=5)
    clusters = km.fit_predict(customer.iloc[:,2:])
    customer["label"] = clusters
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(customer.Age[customer.label == 0], customer["Annual Income (k$)"][customer.label == 0], customer["Spending Score (1-100)"][customer.label == 0], c='green', s=60)
    ax.scatter(customer.Age[customer.label == 1], customer["Annual Income (k$)"][customer.label == 1], customer["Spending Score (1-100)"][customer.label == 1], c='blue', s=60)
    ax.scatter(customer.Age[customer.label == 2], customer["Annual Income (k$)"][customer.label == 2], customer["Spending Score (1-100)"][customer.label == 2], c='yellow', s=60)
    ax.scatter(customer.Age[customer.label == 3], customer["Annual Income (k$)"][customer.label == 3], customer["Spending Score (1-100)"][customer.label == 3], c='red', s=60)
    ax.scatter(customer.Age[customer.label == 4], customer["Annual Income (k$)"][customer.label == 4], customer["Spending Score (1-100)"][customer.label == 4], c='brown', s=60)
    ax.view_init(30, 185)
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    ax.set_zlabel('Spending Score (1-100)')
    ```

<h2>Results</h2>

![image10](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/07/PCA-Cluster-Graph-in-ML.png)

<h4>From the above visualization, we observe that there is a distribution of 6 clusters as follows –</h4>

<h4>Cluster 6 and 4 – These clusters represent the customer_data with the medium income salary as well as the medium annual spend of salary.</h4>

<h4>Cluster 1 – This cluster represents the customer_data having a high annual income as well as a high annual spend.</h4>

<h4>Cluster 3 – This cluster denotes the customer_data with low annual income as well as low yearly spend of income.</h4>

<h4>Cluster 2 – This cluster denotes a high annual income and low yearly spend.</h4>

<h4>Cluster 5 – This cluster represents a low annual income but its high yearly expenditure.</h4>

<h4>Hence using the Kmeans Clustering algorithm we can imbibe a better understanding of target customers which helps in greater profits</h4>






