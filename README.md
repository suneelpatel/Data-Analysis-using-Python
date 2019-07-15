# Data-Analysis_with_Python
Learn to analyze data with Python. Here you will learn, Import data sets, Clean and prepare data for analysis, Manipulate pandas DataFrame, Summarize data, Build machine learning models using scikit-learn, Build data pipelines.

# Table of Content
1. About Data Analysis
2. ETL (Extract, Transform, and Load)
3. Data Manipulation in Python
4. EDA (Exploratory Data Analysis)
5. Statistics Essentials for Analytics
6. Discriptive Statistics
7. Probabilty and It's Uses
8. Statistical Inference
9. Statistical Testing of Data
10. Data Visualization
11. Data Mining
12. Anova and Sentiment Analysis

# 1. About Data Analysis
**Data Analytics** is the science of examining raw data with the purpose of drawing conclusions about that information. It is all about discovering useful information from the data to support decision-making. This process involves inspecting, cleansing, transforming & modeling data.

**Definition**: Data Analysis is a process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, suggesting conclusions, and supporting decision-making.

#### What does a Data Analyst do?
**Data analysts translate numbers into plain English.** Every business collects data, like sales figures, market research, logistics, or transportation costs. A data analyst’s job is to take that data and use it to help companies to make better business decisions.

# 2. ETL (Extract, Transform and Load)
ETL is defined as a process that extracts the data from different RDBMS source systems or other data sources, then transforms the data (like applying calculations, concatenations, etc.) and finally loads the data into the Data Warehouse system. ETL full-form is Extract, Transform and Load.

It's tempting to think a creating a Data warehouse is simply extracting data from multiple sources and loading into database of a Data warehouse. This is far from the truth and requires a complex ETL process. The ETL process requires active inputs from various stakeholders including developers, analysts, testers, top executives and is technically challenging.

### Why do you need ETL?
There are many reasons for adopting ETL in the organization:

* It helps companies to analyze their business data for taking critical business decisions.
* Transactional databases cannot answer complex business questions that can be answered by ETL.
* A Data Warehouse provides a common data repository
* ETL provides a method of moving the data from various sources into a data warehouse.
* As data sources change, the Data Warehouse will automatically update.
* Well-designed and documented ETL system is almost essential to the success of a Data Warehouse project.
* Allow verification of data transformation, aggregation and calculations rules.
* ETL process allows sample data comparison between the source and the target system.
* ETL process can perform complex transformations and requires the extra area to store the data.
* ETL helps to Migrate data into a Data Warehouse. Convert to the various formats and types to adhere to one consistent system.
* ETL is a predefined process for accessing and manipulating source data into the target database.
* ETL offers deep historical context for the business.
* It helps to improve productivity because it codifies and reuses without a need for technical skills.

### Extract – 
Data extraction involves extracting data from homogeneous or heterogeneous sources;

#### Three Data Extraction methods:

* Full Extraction
* Partial Extraction- without update notification.
* Partial Extraction- with update notification

#### Some validations are done during Extraction:

* Reconcile records with the source data
* Make sure that no spam/unwanted data loaded
* Data type check
* Remove all types of duplicate/fragmented data
* Check whether all the keys are in place or not

### Transform – 
Data transformation processes data by data cleansing and transforming them into a proper storage format/structure

#### Validations are done during this stage

* Filtering – Select only certain columns to load
* Using rules and lookup tables for Data standardization
* Character Set Conversion and encoding handling
* Conversion of Units of Measurements like Date Time Conversion, currency conversions, numerical conversions, etc.
* Data threshold validation check. For example, age cannot be more than two digits.
* Data flow validation from the staging area to the intermediate tables.
* Required fields should not be left blank.
* Cleaning ( for example, mapping NULL to 0 or Gender Male to "M" and Female to "F" etc.)
* Split a column into multiples and merging multiple columns into a single column.
* Transposing rows and columns,
* Use lookups to merge data
* Using any complex data validation (e.g., if the first two columns in a row are empty then it automatically reject the row from processing)

### Load – 
Data loading describes the insertion of data into the final target database

#### Types of Loading:

* Initial Load — populating all the Data Warehouse tables
* Incremental Load — applying ongoing changes as when needed periodically.
* Full Refresh —erasing the contents of one or more tables and reloading with fresh data.

#### Load verification
* Ensure that the key field data is neither missing nor null.
* Test modeling views based on the target tables.
* Check that combined values and calculated measures.
* Data checks in dimension table as well as history table.
* Check the BI reports on the loaded fact and dimension table.

#### Key Points of ETL:
* ETL is an abbreviation of Extract, Transform and Load.
* ETL provides a method of moving the data from various sources into a data warehouse.
* In the first step extraction, data is extracted from the source system into the staging area.
* In the transformation step, the data extracted from source is cleansed and transformed .
* Loading data into the target datawarehouse is the last step of the ETL process.

# 3. Data Manipulation in Python
The process of changing data to make it more orgazined and easy to read is known as Data Manipulation.

Before we discuss data manipulation in depth we need to understand **data wrangling or data munging** first.

#### Data Wrangling (Data Munging):
Data Wrangling is the process of converting and mapping data from its raw form to another format with the purpose of making it more valuable and appropriate for advance tasks such as Data Analytics and Machine Learning.

Data wrangling, like most data analytics processes, is an iterative one – the practitioner will need to carry out these steps repeatedly in order to produce the results he desires. There are six broad steps to data wrangling, which are:

##### 1. Discovering
In this step, the data is to be understood more deeply. Before implementing methods to clean it, you will definitely need to have a better idea about what the data is about. Wrangling needs to be done in specific manners, based on some criteria which could demarcate and divide the data accordingly – these are identified in this step.

##### 2. Structuring
Raw data is given to you in a haphazard manner, in most cases – there will not be any structure to it. This needs to be rectified, and the data needs to be restructured in a manner that better suits the analytical method used. Based on the criteria identified in the first step, the data will need to be separated for ease of use. One column may become two, or rows may be split – whatever needs to be done for better analysis.

##### 3. Cleaning
The process of detecting and correcting corrupt or inaccurate records from database or dataset is said to be data cleaning.

All datasets are sure to have some outliers, which can skew the results of the analysis. These will have to be cleaned, for the best results. In this step, the data is cleaned thoroughly for high-quality analysis. Null values will have to be changed, and the formatting will be standardized in order to make the data of higher quality.

Inconsistent and noisy data cannot be used to gain meaningful insights in an organisation. The noisy data needs to be cleaned before it is used for analytical approaches.

We’ll leverage Python’s Pandas and NumPy libraries to clean data.

We’ll cover the following:

* Dropping Columns in a DataFrame
* Changing the Index of a DataFrame
* Tidying up Fields in the Data
* Combining str Methods with NumPy to Clean Columns
* Cleaning the Entire Dataset Using the applymap Function
* Renaming Columns and Skipping Rows

##### 4. Enriching
After cleaning, it will have to be enriched – this is done in the fourth step. This means that you will have to take stock of what is in the data and strategise whether you will have to augment it using some additional data in order to make it better. You should also brainstorm about whether you can derive any new data from the existing clean data set that you have.

##### 5. Validating
Validation rules refer to some repetitive programming steps which are used to verify the consistency, quality and the security of the data you have. For example, you will have to ascertain whether the fields in the data set are accurate via a check across the data, or see whether the attributes are normally distributed.

##### 6. Publishing
The prepared wrangled data is published so that it can be used further down the line – that is its purpose after all. If needed, you will also have to document the steps which were taken or logic used to wrangle the said data.


#### Tidy Data:
##### Tidy data is a standard way of mapping the meaning of a dataset to its structure.

In tidy data:
* 1. Each variable forms a column.
* 2. Each observation forms a row.
* 3. Each type of observational unit forms a table.

#### Impute Missing Values

Imputing relates to applying a model to restore missing values.

There are several options users can consider while replacing a missing value, for example:

* A fixed value that has meaning within the domain, such as 0, distinct from all other values.
* A value from another randomly chosen from the record.
* A mean, median or mode value replaced for the column.
* A value determined by another predictive model.

Any imputing conducted on the training dataset will have to be performed on new data in the future when predictions are required from the finalized model. This needs to be taken into factor when choosing how to impute the missing values.

For example, if one chooses to impute with mean column values, the mean column values will need to be stored to file for later exercise new data that has missing values.


# 4. EDA (Exploratory Data Analysis)

**EDA or Exploratory Data Analysis** is the brainstorming stage of Machine Learning. It is a very important step which takes place after feature engineering and acquiring data and it should be done before any modeling. It's very important for a data scientist to be able to understand the nature of the data without making assumptions.

The purpose of EDA is to use summary statistics and visualizations to better understand data, and find clues about the tendencies (the patterns and trends in the data) of the data, its quality and to formulate assumptions and the hypothesis of our analysis. At this stage, all the useful insights are drawn and correlations between the variables are understood.

For example, in the case of predicting rainfall, we know that there is a strong possibility of rain if the temperature has fallen low. Such correlations must be understood and mapped at this stage.

**EDA or Exploratory Data Analysis** is an approach for summarizing, visualizing, and becoming intimately familiar with the important characteristics of a data set.

It is an approach/philosophy for data analysis that employs a variety of techniques (mostly graphical) to maximize insight into a data set;
* uncover underlying structure;
* extract important variables;
* detect outliers and anomalies;
* test underlying assumptions;
* develop parsimonious models; and
* determine optimal factor settings.

### Value of Exploratory Data Analysis
Exploratory Data Analysis is valuable to data science projects since it allows to get closer to the certainty that the future results will be valid, correctly interpreted, and applicable to the desired business contexts. Such level of certainty can be achieved only after raw data is validated and checked for anomalies, ensuring that the data set was collected without errors. EDA also helps to find insights that were not evident or worth investigating to business stakeholders and data scientists but can be very informative about a particular business.

EDA is performed in order to define and refine the selection of feature variables that will be used for machine learning. Once data scientists become familiar with the data set, they often have to return to feature engineering step, since the initial features may turn out not to be serving their intended purpose. Once the EDA stage is complete, data scientists get a firm feature set they need for supervised and unsupervised machine learning.

### Methods of Exploratory Data Analysis
It is always better to explore each data set using multiple exploratory techniques and compare the results. Once the data set is fully understood, it is quite possible that data scientist will have to go back to data collection and cleansing phases in order to transform the data set according to the desired business outcomes. The goal of this step is to become confident that the data set is ready to be used in a machine learning algorithm.

Exploratory Data Analysis is majorly performed using the following methods:
* **Univariate Visualization** — provides summary statistics for each field in the raw data set
* **Bivariate Visualization** — is performed to find the relationship between each variable in the dataset and the target variable of interest
* **Multivariate Visualization** — is performed to understand interactions between different fields in the dataset
* **Dimensionality Reduction** — helps to understand the fields in the data that account for the most variance between observations and allow for the processing of a reduced volume of data.

Through these methods, the data scientist validates assumptions and identifies patterns that will allow for the understanding of the problem and model selection and validates that the data has been generated in the way it was expected to. So, value distribution of each field is checked, a number of missing values is defined, and the possible ways of replacing them are found.

### Additional benefits Exploratory Data Analysis brings to projects

Another side benefit of EDA is that it allows to specify or even define the questions you are trying to get the answer to from your data. Companies, that are only starting to leverage Data Science and AI technologies, often face the situation when they realize, that they have a lot of data and no ideas of what value that data can bring to their business decision making.
However, the questions always come first in data analysis. It doesn’t matter how much data company has, how many tools they have available, whether the data is historical or real time unless business stakeholders have the questions they are trying to solve with their data. EDA can help such companies to start formalizing the right questions, since with wrong questions you get the wrong answers, and take the wrong decisions.

### Why skipping Exploratory Data Analysis is a bad idea
In a hurry to get to the machine learning stage or simply impress business stakeholders very fast, data scientists tend to either entirely skip the exploratory process or do a very shallow work. It is a very serious and, sadly, common mistake of amateur data science consulting “professionals”.

Such inconsiderate behavior can lead to skewed data, with outliers and too many missing values and, therefore, some sad outcomes for the project:
* generating inaccurate models;
* generating accurate models on the wrong data;
* choosing the wrong variables for the model;
* inefficient use of the resources, including the rebuilding of the model.

It is a classical and under-utilized approach that helps you quickly build a relationship with the new data.

It is always better to explore each data set using multiple exploratory techniques and compare the results. The goal of this step is to understand the dataset, identify the missing values & outliers if any using visual and quantitative methods to get a sense of the story it tells. It suggests the next logical steps, questions or areas of research for your project.

### Steps in Data Exploration and Preprocessing:
1. Identification of variables and data types
2. Analyzing the basic metrics
3. Non-Graphical Univariate Analysis
4. Graphical Univariate Analysis
5. Bivariate Analysis
6. Variable transformations
7. Missing value treatment
8. Outlier treatment
9. Correlation Analysis
10. Dimensionality Reduction


# 5. Statistics Essentials for Analytics
### What is statistics?
Statistics Definition: (Science of Average and their Estimate)
Statistics is the science of collecting, organizing, presenting, analyzing and interpreting data for specific purpose to help in making more effective decision.

#### Why study statistics:
To make more effective decision for the betterment of individual, society, business, nature and so on

#### Statistical Analysis:

Statistical analysis is implemented to manipulate, summarize, and investigate data, so that useful decision making information results are obtained.

##### Two type of statistics:

1. Descriptive Statistics (used to describe the basic features of the data)

2. Inferential statistics (aims at learning characteristics of the population from a sample)

### 1. Data and its Types
#### What is data?
Data is a set of collected or recorded facts of particular subject.

Data in general terms refer to facts and statistics collected together for reference or analysis.

Types of Data:
#### 1. Qualitative Data
#### 2. Quantitative Data 

#### 1. Qualitative Data:
“Data Associated with the quality in different categories”. Data is measurements, each fall into one of several categories. (Hair Color, ethnic groups and other attributes of the population)

##### (a). Nominal Data: “With no inherent order or ranking”
~ Data with no inherent order or ranking such as gender or race, suck kind of data called Nominal Data.

##### (b). Ordinal Data: “with an order series”

#### 2. Quantitative Data: 
“Data associated with Quantity which can be measured”
~ Data measured on a numeric scale (distance travelled to college, the number of children in a family etc.)

##### (a). Discrete Data: “Based on count, finite number of values possible and value cannot be subdivided”
~ Data which can be categorized into classification, data which is based upon counts, there are only a finite number of values possible and values cannot be subdivided meaningfully, such kind of data is called Discrete Data.

##### (b). # Continuous Data: “measured on a continuum or a scale, value which can be subdivided into finer increments”
~ Data which can be measured on a continuum or a scale, data which can be have almost any numeric value and can be subdivided into finer and finer increments, such kind of data is called Continuous Data.

### 2. Variable and it's Types
#### What is variable?
A variable in algebra represents an unknown value or a value that varies.

#### Types of Variables:
##### 1. Categorical Variable:
Variable that can be put into categories. For example, male and female are two categories in a Gender.

##### 2. Control Variable:
A factor in an experiment which must be held constant

##### 3. Confounding Variable:
Extra variables that have a hidden effects on your experimental results

##### 4. Dependent Variable (Output Variable):
The outcome of an experiment

##### 5. Independent Variable (Input Variable):
A variable that is not affected by anything


### 3. Sample and Population
#### Population:
A Population is the set of all possible states of a random variable. The size of the population may be either infinite or finite.

In other words, A collection or set of individual or objects or events whose properties are to be anlysed called population.

#### Sample:
A Sample is a subset of the population; its size is always finite.

A subset of population is called "Sample". A well choosen sample will contain most of the information about a particular population parameter.


### 4. Sampling Techniques

#### 1. Probability Sampling : 
This Sampling technique uses randomization to make sure that every element of the population gets an equal chance to be part of the selected sample. It’s alternatively known as random sampling.

##### (a) Random Sampling : 
Every element has an equal chance of getting selected to be the part sample. It is used when we don’t have any kind of prior information about the target population.

For example: Random selection of 20 students from class of 50 student. Each student has equal chance of getting selected. Here probability of selection is 1/50.

##### (b) Systematic Sampling 

Here the selection of elements is systematic and not random except the first element. Elements of a sample are chosen at regular intervals of population. All the elements are put together in a sequence first where each element has the equal chance of being selected.

For a sample of size n, we divide our population of size N into subgroups of k elements.

We select our first element randomly from the first subgroup of k elements.

To select other elements of sample, perform following:

We know number of elements in each group is k i.e N/n

So if our first element is n1 then

Second element is n1+k i.e n2

Third element n2+k i.e n3 and so on..

Taking an example of N=20, n=5

No of elements in each of the subgroups is N/n i.e 20/5 =4= k

Now, randomly select first element from the first subgroup.

If we select n1= 3

n2 = n1+k = 3+4 = 7

n3 = n2+k = 7+4 = 11


##### (c) Stratified Sampling 
This technique divides the elements of the population into small subgroups (strata) based on the similarity in such a way that the elements within the group are homogeneous and heterogeneous among the other subgroups formed. And then the elements are randomly selected from each of these strata. We need to have prior information about the population to create subgroups.

#### 2. Non-Probability Sampling : 
It does not rely on randomization. This technique is more reliant on the researcher’s ability to select elements for a sample. Outcome of sampling might be biased and makes difficult for all the elements of population to be part of the sample equally. This type of sampling is also known as non-random sampling.

##### (a) Snowball Sampling:
This technique is used in the situations where the population is completely unknown and rare.
Therefore we will take the help from the first element which we select for the population and ask him to recommend other elements who will fit the description of the sample needed.

So this referral technique goes on, increasing the size of population like a snowball.

##### For example: 
It’s used in situations of highly sensitive topics like HIV Aids where people will not openly discuss and participate in surveys to share information about HIV Aids.

Not all the victims will respond to the questions asked so researchers can contact people they know or volunteers to get in touch with the victims and collect information

Helps in situations where we do not have the access to sufficient people with the characteristics we are seeking. It starts with finding people to study.

##### (b) Quota Sampling: 
This type of sampling depends of some pre-set standard. It selects the representative sample from the population. Proportion of characteristics/ trait in sample should be same as population. Elements are selected until exact proportions of certain types of data is obtained or sufficient data in different categories is collected.

##### For example: 
If our population has 45% females and 55% males then our sample should reflect the same percentage of males and females.

##### (c) Judgement sampling 
This is based on the intention or the purpose of study. Only those elements will be selected from the population which suits the best for the purpose of our study.

##### For Example: 
If we want to understand the thought process of the people who are interested in pursuing master’s degree then the selection criteria would be “Are you interested for Masters in..?”

All the people who respond with a “No” will be excluded from our sample.

##### (d) Convenience Sampling
Here the samples are selected based on the availability. This method is used when the availability of sample is rare and also costly. So based on the convenience samples are selected.

##### For example: 
Researchers prefer this during the initial stages of survey research, as it’s quick and easy to deliver results.


# 6. Descriptive Statistics: 
Collecting, Summarizing or Describing and Processing data to transform data into information

Descriptive statistics are used to describe the basic features of the data in a study.
* Descriptive statistics is a data analysis strategy.
* It deals with the representation of numerical facts, or data, in either table or graphic form, and with the methodology of analysis the data.

Example: A student’s grade point average (GPA), provides a good understanding in analysing his overall performance.


### Type of Descriptive Statstics:
Descriptive statistics are broken down into two categories. **Measures of Central Tendency** and **Measures of Spread (variability or dispersion)**.

### (1) Measure of Centre(Central Tendency):
The data values for most numerical variables tend to group around a specific value

Measure of centre help us to describe what extent this pattern holds for a specific numerical variable

Three commonly-used measures of centre:

##### (a) Mean (also known as the arithmetic mean or average)
##### (b) Median  
##### (c) Mode

#### Mean: “An Average”
The mean (or average) of a number of observations is the sum of the values of all the observations divided by the total number of observations. It is denoted by the symbol X, read as ‘X bar’.

#### Median: “A middle Value”
The median is that value of the given number of observations, which divides it into exactly two parts. So, when the data is arranged in ascending (or descending) order the median of ungrouped data is calculated as follows:

(i) When the number of observations (n) is odd, the median is the value of the {(n+1)/2}th observation. For example, if n = 13, the value of the {(13+1)/2}th, i.e., the 7th observation will be the median.

(ii) When the number of observations (n) is even, the median is the mean of the {n/2}th and the {(n+1)/2}th observations.

#### Mode: “The highest or maximum number of frequency”
The mode is the most common observation of a data set, or the value in the data set that occurs most frequently.

#### Comparison between median and mean:
Median:
•	Ignore the extreme value
•	Tell the point from where 50% data is lesser and 50% is more

Mean:
•	All the data are given equal importance

#### Relationship among all
Mean – Mode = 3 (Mean - Median)

Mode = 3Median – 2Mean 


### (2) Measure of Spread (Variability / Dispersion) 
A measure of spread, sometimes also called a measure of dispersion or measure of variability is used to describe the variability in a sample or population.

It is usually used in conjunction with measure of central tendency, such as the mean or median, to provide an overall description of a set of data.

##### (a) Range 
##### (b) Percentiles/Quartiles
##### (c) Inter-Quartile Range (IQR) 
##### (d) Variance
##### (e) Standard Deviation
##### (f) Skewness
##### (g) Kurtosis

#### (a) Range: 

The range is simply the difference between the maximum and minimum values in a data set. 
##### Range = max - min

So in a data set of 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 11, 13, 15, 15, 17, 19, 20, the range is the difference between 2 and 20.
18 = 20 - 2

While it is useful in seeing how large the difference in observations is in a sample, it says nothing about the spread of the data.

#### (b) Percentiles/Quartiles
##### Percentiles divide a data set into 100 equal parts. A percentile is simply a measure that tells us what percent of the total frequency of a data set was at or below that measure.

##### The Quartiles also divide the data into divisions of 25%, so:

Quartile 1 (Q1) can be called the 25th percentile
Quartile 2 (Q2) can be called the 50th percentile
Quartile 3 (Q3) can be called the 75th percentile

#### (c) Inter-Quartile Range (IQR) 
The inter-quartile range (IQR) gives more information about how the observation values of a data set are dispersed. It shows the range of the middle 50% of observations.

#### (d) Variance

##### Deviation: The difference between each xi and the mean is called deviation about the mean

##### Variance: is based on deviations and entails computing square of deviations

##### Population Variance: Average of Standard Deviations

##### Sample Variance: sum of square deviations divided by n-1

#### (e) Standard Deviation

The standard deviation indicates the average distance between an observation value, and the mean of a data set. In this way, it shows how well the mean represents the values in a data set. Like the mean, it is appropriate to use when the data set is not skewed or containing outliers.

#### (f) Skewness
In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive or negative, or undefined.

#### (g) Kurtosis
In probability theory and statistics, kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.


## Information Gain and Entropy

### Information Gain:
* Information is a measure of facts
* Information gain is the ratio of factual information, to uncertain information
* Is signifies a reduction is entropy or uncertain

### Entropy:
#### What is Entropy? 
* In the most layman terms, Entropy is nothing but the measure of disorder or uncertainty (You can think of it as a measure of purity as well) the goal of machine learning models and Data Scientists in general is to reduce uncertainty.
* We sometimes need to choose the options which have high information gain and low entropy while taking crucial decision
##### Example: We are certain that flight 3 p.m., but uncertain regarding the exact time to reach airport 


##### Confusion Matrix:
A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on set of test data for which the true value are known:
* Confusion matrix represents a tabular presentation of Actual Vs Predict Value
* You can calculate the accuracy of your model with:

          (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)

#### Example: 
* There are two possible predicted classes: “yes” and “no”
* The classifier made a total of 165 predictions
* Out of those 165 cases, the classifier predicted “yes” 110 times, and “no” 55 times
* In reality, 105 patients in the sample have the disease, and 60 patient do not

##### (A): Type I Error: We predict yes, but they don’t actually have the disease (Also known as Type I Error)

##### (B): Type II Error: We predict No, but they actually do have the disease (Also known as Type II Error)


### Sensitivity:
* Sensitivity (also called the true positive rate, the recall, or probability of detection in some fields) measures the proportion of positives that are correctly identified
* In probability notation:
                              
                              Sensitivity = TRUE POSITIVE / (TRUE POSITIVE + FALSE NEGATIVE)


### Specificity:
* Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such (e.g. the percentage of healthy people who are correctly identified as not having the condition)
* In probability notation:

                              Specificity = TRUE NEGATIVE / (TRUE NEGATIVE + FALSE POSITIVE)
                              

# 7. Probability and it's Uses

### What is probability?	

A measure of uncertainty of various phenomenon, numerically

* Probability is measure of how likely something will occur
* It is the ratio of desired outcomes to total outcomes: 

                    Desired Outcomes/ #Total Outcomes

##### In mathematical terms :
P (E) = No. of favourable outcome / Total no. of outcomes

##### The probability of all outcomes always sum of 1
P (E) + P (E’) = 1

### Terminologies of Probability
##### (1) Random Experiment:
An operation which can produce some well-defined outcomes is called an experiment. Each outcome is called an event

For example; throwing a die or tossing a coin etc.
In an experiment where all possible outcomes are known and in advance if the exact outcome cannot be predicted, is called a random experiment. 

##### (3) Trial:
By a trial, we mean performing a random experiment.

##### (4) Sample Space 
The sample space for a probability experiment is the set of all possible outcomes. This is usually written with set notation (curly brackets). For example, going back to a regular 6-sided die the sample space would be:

S={1,2,3,4,5,6}

##### (5) Event 
Out of the total results obtained from a certain experiment, the set of those results which are in favor of a definite result is called the event and it is denoted as E.

#### (6) Equally Likely Events:

When there is no reason to expect the happening of one event in preference to the other, then the events are known equally likely events.

For example; when an unbiased coin is tossed the chances of getting a head or a tail are the same.

#### (7) Exhaustive Events:

All the possible outcomes of the experiments are known as exhaustive events.

For example; in throwing a die there are 6 exhaustive events in a trial.

#### (8) Favorable Events:

The outcomes which make necessary the happening of an event in a trial are called favorable events.

For example; if two dice are thrown, the number of favorable events of getting a sum 5 is four,

i.e., (1, 4), (2, 3), (3, 2) and (4, 1).

#### (9) Mutually Exclusive Events: 
If there be no element common between two or more events, i.e., between two or more subsets of the sample space, then these events are called mutually exclusive events.

If E1 and E2 are two mutually exclusive events, then E1 ∩ E2 = ∅

#### (10) Complementary Event: 
An event which consists in the negation of another event is called complementary event of the er event. In case of throwing a die, ‘even face’ and ‘odd face’ are complementary to each other. “Multiple of 3” ant “Not multiple of 3” are complementary events of each other.

##### (11) Union of Event:
Union of events is simply a union of two or more than two events. If A and B are two events then A U B is called union of A and B. suppose that two events are given A and B then. The union of two events A and B is the event which consists all the elements of A and B.

##### Formula:
Suppose A and B are two events associated with a random experiment. Then the union of A and B is represented by A ∪ B.

The probability of union of two events is given by:

P(A∪B) = P(A)+P(B) – P(A∩B)

Here, P (A) is the probability of event A, P (B) is the probability of event B.

Also, P(A∩B) is the probability of the intersection of events A and B. 


When A and B are two independent or mutually exclusive events that is the occurrence of event A does not affect the occurrence of event B at all, in such a case, P(A∩B) = 0 and hence we have,

P(A∪B) = P(A)+P(B)

If we have more than two independent events say A, B & C, then in that case the union probability is given by:

P(A∪B∪C) = P(A)+P(B)+P(C)

If AB and C are not independent or mutually exclusive then the union probability is given by:

P(A∪B∪C) = P(A)+P(B)+P(C) – P(A∩B)–P(B∩C)–P(A∩C) - P(A∩B∩C)


##### (12) Intersection of Event
Intersection of events means that all the events are occurring together. Even if one event holds false all will be false. The intersection of events can only be true if and only if all the events holds true. 

The probability that Events A and B both occur is the probability of the intersection of A and B. The probability of the intersection of Events A and B is denoted by P(A ∩ B). If Events A and B are mutually exclusive, P(A ∩ B) = 0.

The rule of multiplication applies to the situation when we want to know the probability of the intersection of two events; that is, we want to know the probability that two events (Event A and Event B) both occur.

Rule of Multiplication The probability that Events A and B both occur is equal to the probability that Event A occurs times the probability that Event B occurs, given that A has occurred.

P(A ∩ B) = P(A) P(B|A)

## Probability Distribution:
Probability is often associated with at least one event. This event can be anything. Basic examples of events include rolling a die or pulling a coloured ball out of a bag. In these examples the outcome of the event is random (you can’t be sure of the value that the die will show when you roll it), so the **variable that represents the outcome of these events is called a random variable (often abbreviated to RV)**.

### The 3 types of probability

#### Marginal Probability: 
If A is an event, then the marginal probability is the probability of that event occurring, P(A). 
**Example:** Assuming that we have a pack of traditional playing cards, an example of a marginal probability would be the probability that a card drawn from a pack is red: P(red) = 0.5.

#### Joint Probability: 
The probability of the intersection of two or more events. Visually it is the intersection of the circles of two events on a Venn Diagram (see figure below). If A and B are two events then the joint probability of the two events is written as P(A ∩ B). 
**Example:** The probability that a card drawn from a pack is red and has the value 4 is P(red and 4) = 2/52 = 1/26. (There are 52 cards in a pack of traditional playing cards and the 2 red ones are the hearts and diamonds). We’ll go through this example in more detail later.

#### Conditional Probability: 
The conditional probability is the probability that some event(s) occur given that we know other events have already occurred. If A and B are two events then the conditional probability of A occurring given that B has occurred is written as P(A|B). 
**Example:** The probability that a card is a four given that we have drawn a red card is P(4|red) = 2/26 = 1/13. (There are 52 cards in the pack, 26 are red and 26 are black. Now because we’ve already picked a red card, we know that there are only 26 cards to choose from, hence why the first denominator is 26).

##### Distribution:
Before we jump on to the explanation of distributions, let’s see what kind of data can we encounter. The data can be discrete or continuous.

**Discrete Data**, as the name suggests, can take only specified values. For example, when you roll a die, the possible outcomes are 1, 2, 3, 4, 5 or 6 and not 1.5 or 2.45.

**Continuous Data** can take any value within a given range. The range may be finite or infinite. For example, A girl’s weight or height, the length of the road. The weight of a girl can be any value from 54 kgs, or 54.5 kgs, or 54.5436kgs.

##### Types of Distributions
**1.Bernoulli Distribution**
**2. Uniform Distribution**
**3. Binomial Distribution**
**4. Normal Distribution**
**5. Poisson Distribution**
**6. Exponential Distribution**

#### Normal Distribution
Normal distribution represents the behavior of most of the situations in the universe (That is why it’s called a “normal” distribution). The large sum of (small) random variables often turns out to be normally distributed, contributing to its widespread application. Any distribution is known as Normal distribution if it has the following characteristics:

* The mean, median and mode of the distribution coincide.
* The curve of the distribution is bell-shaped and symmetrical about the line x=μ.
* The total area under the curve is 1.
* Exactly half of the values are to the left of the center and the other half to the right.
* A normal distribution is highly different from Binomial Distribution. However, if the number of trials approaches infinity then the shapes will be quite similar.

##### Some of the properties of a standard normal distribution are mentioned below:
* The normal curve is symmetric about the mean and bell shaped.
* Mean, mode and median is zero which is the centre of the curve.
* Approximately 68% of the data will be between -1 and +1 (i.e. within 1 standard deviation from the mean), 95% between -2 and +2 (within 2 SD from the mean) and 99.7% between -3 and 3 (within 3 SD from the mean)

##### There are a few commonly used terms which we need to understand:
* **Population** : Space of all possible elements from a set of data
* **Sample** : consists of observations drawn from population
* **Parameter** : measurable characteristic of a population such as mean, SD
* **Statistic**: measurable characteristic of a sample


#3# Central Limit Theorem (CLT)
The central limit theorem (CLT) is simple. It just says that with a large sample size, sample means are normally distributed.

Well, the central limit theorem (CLT) is at the heart of hypothesis testing – a critical component of the data science lifecycle.

#### Formally Defining the Central Limit Theorem:
Given a dataset with unknown distribution (it could be uniform, binomial or completely random), the sample means will approximate the normal distribution.

#### Assumptions Behind the Central Limit Theorem
Before we dive into the implementation of the central limit theorem, it’s important to understand the assumptions behind this technique:

* The data must follow the **randomization condition**. It must be sampled randomly
* Samples should be **independent of each other**. One sample should not influence the other samples
* **Sample size** should be not more than 10% of the population when sampling is done without replacement
* The sample size should be sufficiently large. Now, how we will figure out how large this size should be? Well, it depends on the population. When the population is skewed or asymmetric, the sample size should be large. If the population is symmetric, then we can draw small samples as well
In general, a **sample size of 30 is considered sufficient when the population is symmetric**.

          The mean of the sample means is denoted as:

                    µ X̄ = µ

          where,

          µ X̄ = Mean of the sample means
          µ= Population mean
          
          
          And, the standard deviation of the sample mean is denoted as:

                    σ X̄ = σ/sqrt(n)

          where,

          σ X̄ = Standard deviation of the sample mean
          σ = Population standard deviation
          n = sample size


### Baye's Theorem (aka, Bayes Rule)
Before understanding Baye's Theorem first we learn about **Conditional Probability**:

##### Conditional Probability : 
The probability that event A occurs, given that event B has occurred, is called a conditional probability.

The conditional probability of A, given B, is denoted by the symbol P(A|B).

##### Baye's Theorem:
* Bayes' theorem (also known as Bayes' rule) is a useful tool for calculating conditional probabilities. 

* Bayes’ Theorem is a way of finding a probability when we know certain other probabilities.

Bayes' theorem can be stated as follows:

                           
                    The formula is: P(A|B) = P(A) P(B|A)P(B) 

          Which tells us:     how often A happens given that B happens, written P(A|B),
          When we know:       how often B happens given that A happens, written P(B|A),
 	 	          and how likely A is on its own, written P(A),
 	 	          and how likely B is on its own, written P(B)

#### Bayes Theorem Rule:

The rule has a very simple derivation that directly leads from the relationship between joint and conditional probabilities. First, note that P(A,B) = P(A|B)P(B) = P(B,A) = P(B|A)P(A). Next, we can set the two terms involving conditional probabilities equal to each other, so P(A|B)P(B) = P(B|A)P(A), and finally, divide both sides by P(B) to arrive at Bayes rule.

In this formula, A is the event we want the probability of, and B is the new evidence that is related to A in some way.

P(A|B) is called the **posterior**; this is what we are trying to estimate. In the above example, this would be the “probability of having cancer given that the person is a smoker”.

P(B|A) is called the **likelihood**; this is the probability of observing the new evidence, given our initial hypothesis. In the above example, this would be the “probability of being a smoker given that the person has cancer”.

P(A) is called the **prior**; this is the probability of our hypothesis without any additional prior information. In the above example, this would be the “probability of having cancer”.

P(B) is called the marginal **likelihood**; this is the total probability of observing the evidence. In the above example, this would be the “probability of being a smoker”. In many applications of Bayes Rule, this is ignored, as it mainly serves as normalization.

##### Example:
Let us say P(Fire) means how often there is fire, and P(Smoke) means how often we see smoke, then:

P(Fire|Smoke) means how often there is fire when we can see smoke 
P(Smoke|Fire) means how often we can see smoke when there is fire

So the formula kind of tells us "forwards" P(Fire|Smoke) when we know "backwards" P(Smoke|Fire)

          Example: If dangerous fires are rare (1%) but smoke is fairly common (10%) due to barbecues, and 90% of dangerous fires make smoke then:
          P(Fire|Smoke) = P(Fire) * P(Smoke|Fire) / P(Smoke) 
                        = 1% x 90% / 10% 
                        = 9%
          So the "Probability of dangerous Fire when there is Smoke" is 9%

# 8. Statistical Inference
### What is statistical inference?
Statistical inference is the process of drawing conclusions about populations or scientific truths from data.

The four-step process that encompasses statistics: Data Production, Exploratory Data Analysis, Probability and Inference.

A **statistical inference** aims at learning characteristics of the population from a sample; the population characteristics are parameters and sample characteristics are statistics.

A **statistical model** is a representation of a complex phenomena that generated the data.
* It has mathematical formulations that describe relationships between random variables and parameters.
* It makes assumptions about the random variables, and sometimes parameters.
* A general form: data = model + residuals
* Model should explain most of the variation in the data
* Residuals are a representation of a lack-of-fit, that is of the portion of the data unexplained by the model.

**Estimation** represents ways or a process of learning and determining the population parameter based on the model fitted to the data.

**Point Estimation** and **Interval Estimation**, and **Hypothesis Testing** are three main ways of learning about the population parameter from the sample statistic.

An **estimator** is particular example of a statistic, which becomes an **estimate** when the formula is replaced with actual observed sample values.

**Point Estimation** = a single value that estimates the parameter. Point estimates are single values calculated from the sample

**Confidence Intervals** = gives a range of values for the parameter Interval estimates are intervals within which the parameter is expected to fall, with a certain degree of confidence.

**Hypothesis Tests** = tests for a specific value(s) of the parameter.
In order to perform these inferential tasks, i.e., make inference about the unknown population parameter from the sample statistic, we need to know the likely values of the sample statistic. What would happen if we do sampling many times?

We need the sampling distribution of the statistic
* It depends on the model assumptions about the population distribution, and/or on the sample size.
* Standard error refers to the standard deviation of a sampling distribution.


## Hypothesis Testing
### What is Hypothesis Testing?
A statistical hypothesis is an assumption about a population parameter. This assumption may or may not be true. Hypothesis testing refers to the formal procedures used by statisticians to accept or reject statistical hypotheses.

#### Statistical Hypotheses
The best way to determine whether a statistical hypothesis is true would be to examine the entire population. Since that is often impractical, researchers typically examine a random sample from the population. If sample data are not consistent with the statistical hypothesis, the hypothesis is rejected.

#### There are two types of statistical hypotheses.

#### Null hypothesis. 
The null hypothesis, denoted by Ho, is usually the hypothesis that sample observations result purely from chance.
#### Alternative hypothesis. 
The alternative hypothesis, denoted by H1 or Ha, is the hypothesis that sample observations are influenced by some non-random cause.

### Hypothesis Tests
Statisticians follow a formal process to determine whether to reject a null hypothesis, based on sample data. This process, called hypothesis testing, consists of four steps.

* **State the hypotheses**: This involves stating the null and alternative hypotheses. The hypotheses are stated in such a way that they are mutually exclusive. That is, if one is true, the other must be false.
* **Formulate an analysis plan**: The analysis plan describes how to use sample data to evaluate the null hypothesis. The evaluation often focuses around a single test statistic.
* **Analyze sample data**: Find the value of the test statistic (mean score, proportion, t statistic, z-score, etc.) described in the analysis plan.
* **Interpret results**: Apply the decision rule described in the analysis plan. If the value of the test statistic is unlikely, based on the null hypothesis, reject the null hypothesis.

### Decision Errors
Two types of errors can result from a hypothesis test.

#### Type I error. 
A Type I error occurs when the researcher rejects a null hypothesis when it is true. The probability of committing a Type I error is called the significance level. This probability is also called alpha, and is often denoted by α.

#### Type II error. 
A Type II error occurs when the researcher fails to reject a null hypothesis that is false. The probability of committing a Type II error is called Beta, and is often denoted by β. The probability of not committing a Type II error is called the Power of the test.

### Decision Rules
The analysis plan includes decision rules for rejecting the null hypothesis. In practice, statisticians describe these decision rules in two ways - with reference to a P-value or with reference to a region of acceptance.

* **P-value**. The strength of evidence in support of a null hypothesis is measured by the P-value. Suppose the test statistic is equal to S. The P-value is the probability of observing a test statistic as extreme as S, assuming the null hypothesis is true. If the P-value is less than the significance level, we reject the null hypothesis.
* **Region of acceptance**. The region of acceptance is a range of values. If the test statistic falls within the region of acceptance, the null hypothesis is not rejected. The region of acceptance is defined so that the chance of making a Type I error is equal to the significance level.
* The set of values outside the region of acceptance is called the region of rejection. If the test statistic falls within the region of rejection, the null hypothesis is rejected. In such cases, we say that the hypothesis has been rejected at the α level of significance.

#### Significance Level 
Significance level is the probablity of rejecting the null hypothesis when it is true, which is known as **Type I Error**. Denoted by alpha.

#### Confidence Level 
The Confidence level is just the compliment of Significance level which signifies how confident you are in your decision. Express as 1 - alpha.

Confidence Level + Significance Level = 1 (always)

#### Computing the Significance Level : Two ways the significance level can be calculated:

##### (A) One Tail Test :
One-Tailed and Two-Tailed Tests
A test of a statistical hypothesis, where the region of rejection is on only one side of the sampling distribution, is called a **one-tailed test**. 

**For example**, suppose the null hypothesis states that the mean is less than or equal to 10. The alternative hypothesis would be that the mean is greater than 10. The region of rejection would consist of a range of numbers located on the right side of sampling distribution; that is, a set of numbers greater than 10.

Example :- a college has ≥ 4000 student or data science ≤ 80% org adopted.

##### (2) Two Tail Test
A test of a statistical hypothesis, where the region of rejection is on both sides of the sampling distribution, is called a **two-tailed test**. 

**For example**, suppose the null hypothesis states that the mean is equal to 10. The alternative hypothesis would be that the mean is less than 10 or greater than 10. The region of rejection would consist of a range of numbers located on both sides of sampling distribution; that is, the region of rejection would consist partly of numbers that were less than 10 and partly of numbers that were greater than 10.

Example : a college != 4000 student or data science != 80% org adopted


# 9. Statistical Testing of Data

Statistical Tests are intended to decide weather a hypothesis about distribution of one or more populations should be accepted or rejected.

Their are two type of statistical tests:
#### (1) Parametric Tests
#### (2) Non Parametric Tests

#### Why to use Statistical Testing?
* To calculate the difference in the sample and population means
* To find the difference in sample means
* To test the significance of association between two variables
* To calculate several population means
* To test the difference in proportions between two independent populations
* To test the difference in proporation between sample and population

#### What are parameters?
* Parameters are numbers which summarize the data for the entrire population, while statistics are numbers which summarize the data from a sample
* Parametric Testing is used for quanititve data and continuous variables

#### (1) Parametric Tests : A parametric test makes assumption regarding population parameters and distribution
##### (a) Z Testing
##### (b) Student T-Testing
##### (c) P Testing
##### (d) ANOVA Testing

#### (a) Z Testing:
The Z Test is used for testing significance difference between two point estimates
##### Assumptions for Z Test
* The sample must be randomly selected and data must be quantitative
* Sample should be larger
* Data should follow a normal distribution

#### (2) Non-Parametric Tests:
* Chi-Square Testing

### A/B Testing:


##### Problem 1: Two-Tailed Test

The CEO of a large electric utility claims that 80 percent of his 1,000,000 customers are very satisfied with the service they receive. To test this claim, the local newspaper surveyed 100 customers, using simple random sampling. Among the sampled customers, 73 percent say they are very satisified. Based on these findings, can we reject the CEO's hypothesis that 80% of the customers are very satisfied? Use a 0.05 level of significance.

##### Solution: 
The solution to this problem takes four steps: (1) state the hypotheses, (2) formulate an analysis plan, (3) analyze sample data, and (4) interpret results. We work through those steps below:

State the hypotheses. The first step is to state the null hypothesis and an alternative hypothesis.

Null hypothesis: P = 0.80

Alternative hypothesis: P ≠ 0.80

Note that these hypotheses constitute a two-tailed test. The null hypothesis will be rejected if the sample proportion is too big or if it is too small.

Formulate an analysis plan. For this analysis, the significance level is 0.05. The test method, shown in the next section, is a one-sample z-test.

Analyze sample data. Using sample data, we calculate the standard deviation (σ) and compute the z-score test statistic (z).

          σ = sqrt[ P * ( 1 - P ) / n ]

          σ = sqrt [(0.8 * 0.2) / 100]

          σ = sqrt(0.0016) = 0.04

          z = (p - P) / σ = (.73 - .80)/0.04 = -1.75

          where P is the hypothesized value of population proportion in the null hypothesis, p is the sample proportion, and n is the sample size.

Since we have a two-tailed test, the P-value is the probability that the z-score is less than -1.75 or greater than 1.75.

We use the Normal Distribution Calculator to find P(z < -1.75) = 0.04, and P(z > 1.75) = 0.04. Thus, the P-value = 0.04 + 0.04 = 0.08.
Interpret results. Since the P-value (0.08) is greater than the significance level (0.05), we cannot reject the null hypothesis.
Note: If you use this approach on an exam, you may also want to mention why this approach is appropriate. Specifically, the approach is appropriate because the sampling method was simple random sampling, the sample included at least 10 successes and 10 failures, and the population size was at least 10 times the sample size.


##### Problem 2: One-Tailed Test
Suppose the previous example is stated a little bit differently. Suppose the CEO claims that at least 80 percent of the company's 1,000,000 customers are very satisfied. Again, 100 customers are surveyed using simple random sampling. The result: 73 percent are very satisfied. Based on these results, should we accept or reject the CEO's hypothesis? Assume a significance level of 0.05.

##### Solution: 
The solution to this problem takes four steps: (1) state the hypotheses, (2) formulate an analysis plan, (3) analyze sample data, and (4) interpret results. We work through those steps below:

State the hypotheses. The first step is to state the null hypothesis and an alternative hypothesis.

Null hypothesis: P >= 0.80

Alternative hypothesis: P < 0.80

Note that these hypotheses constitute a one-tailed test. The null hypothesis will be rejected only if the sample proportion is too small.

Formulate an analysis plan. For this analysis, the significance level is 0.05. The test method, shown in the next section, is a one-sample z-test.

Analyze sample data. Using sample data, we calculate the standard deviation (σ) and compute the z-score test statistic (z).
          
          σ = sqrt[ P * ( 1 - P ) / n ] = sqrt [(0.8 * 0.2) / 100]

          σ = sqrt(0.0016) = 0.04

          z = (p - P) / σ = (.73 - .80)/0.04 = -1.75

          where P is the hypothesized value of population proportion in the null hypothesis, p is the sample proportion, and n is the sample size.

Since we have a one-tailed test, the P-value is the probability that the z-score is less than -1.75. We use the Normal Distribution Calculator to find P(z < -1.75) = 0.04. Thus, the P-value = 0.04.
Interpret results. Since the P-value (0.04) is less than the significance level (0.05), we cannot accept the null hypothesis.
Note: If you use this approach on an exam, you may also want to mention why this approach is appropriate. Specifically, the approach is appropriate because the sampling method was simple random sampling, the sample included at least 10 successes and 10 failures, and the population size was at least 10 times the sample size.


# 10. Data Visualization

**Data visualization** is the graphical or pictorial representation of information and data.

Python has already made it easy for you – with two exclusive libraries for visualization, commonly known as matplotlib and seaborn.
* Matplotlib
* Seaborn

### Matplotlib (Data Visualization)

**Matplotlib** is a Python library this is specially desigend for the development of graphs, charts etc., in order to provide interactive data visualization.

Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

Matplotlib tries to make easy things easy and hard things possible. You can generate plots, histograms, power spectra, bar charts, errorcharts, scatterplots, etc., with just a few lines of code.

Matplotlib is inspried form the MATLAB software and reproduces many of it's feature

### Seaborn: 
**Seaborn** is a library for creating informative and attractive statistical graphics in python. This library is based on matplotlib. 

Seaborn offers various features such as built in themes, color palettes, functions and tools to visualize univariate, bivariate, linear regression, matrices of data, statistical time series etc which lets us to build complex visualizations.

Its standard designs are awesome and it also has a nice interface for working with pandas dataframes.
 
Data visualization is the discipline of trying to understand data by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be exposed.


# 11. Data Mining
Data mining is the process of analysing **vast amounts of data** from various sources to extract useful information. This is done through the discovery of previously unknown patterns, correlations, and anomalies, which can then be used to predict future outcomes.

Data mining is also called as Knowledge discovery, Knowledge extraction, data/pattern analysis, information harvesting, etc.

**Machine learning**, takes things further by using algorithms and an iterative process to learn from new data and automatically become better at analysis and prediction. It can do this without the need for human intervention.

It is a multi-disciplinary skill that uses machine learning, statistics, AI and database technology.

The insights derived via Data Mining can be used for marketing, fraud detection, and scientific discovery, etc.

### Perform Data Mining: Types of Data
Data mining can be performed on following types of data

* Relational databases
* Data warehouses
* Advanced DB and information repositories
* Object-oriented and object-relational databases
* Transactional and Spatial databases
* Heterogeneous and legacy databases
* Multimedia and streaming database
* Text databases
* Text mining and Web mining

### Data Mining Implementation Process
#### Step1: Business Understanding
#### Step2: Data Understanding
#### Step3: Data Preparation
#### Step4: Data Transformation
#### Step5: Modelling
#### Step6: Evalution
#### Step7: Deployment

Let's study the Data Mining implementation process in detail

#### Business understanding:
In this phase, business and data-mining goals are established.

* First, you need to understand business and client objectives. You need to define what your client wants (which many times even they do not know themselves)
* Take stock of the current data mining scenario. Factor in resources, assumption, constraints, and other significant factors into your assessment.
* Using business objectives and current scenario, define your data mining goals.
* A good data mining plan is very detailed and should be developed to accomplish both business and data mining goals.

#### Data understanding:
In this phase, sanity check on data is performed to check whether its appropriate for the data mining goals.

* First, data is collected from multiple data sources available in the organization.
* These data sources may include multiple databases, flat filer or data cubes. There are issues like object matching and schema integration which can arise during Data Integration process. It is a quite complex and tricky process as data from various sources unlikely to match easily. For example, table A contains an entity named cust_no whereas another table B contains an entity named cust-id.
* Therefore, it is quite difficult to ensure that both of these given objects refer to the same value or not. Here, Metadata should be used to reduce errors in the data integration process.
* Next, the step is to search for properties of acquired data. A good way to explore the data is to answer the data mining questions (decided in business phase) using the query, reporting, and visualization tools.
* Based on the results of query, the data quality should be ascertained. Missing data if any should be acquired.

#### Data preparation:
In this phase, data is made production ready.

* The data preparation process consumes about 90% of the time of the project.
* The data from different sources should be selected, cleaned, transformed, formatted, anonymized, and constructed (if required).
* Data cleaning is a process to "clean" the data by smoothing noisy data and filling in missing values.

For example, for a customer demographics profile, age data is missing. The data is incomplete and should be filled. In some cases, there could be data outliers. For instance, age has a value 300. Data could be inconsistent. For instance, name of the customer is different in different tables.

* Data transformation operations change the data to make it useful in data mining. Following transformation can be applied

#### Data transformation:
Data transformation operations would contribute toward the success of the mining process.

**Smoothing:** It helps to remove noise from the data.

**Aggregation:** Summary or aggregation operations are applied to the data. I.e., the weekly sales data is aggregated to calculate the monthly and yearly total.

**Generalization:** In this step, Low-level data is replaced by higher-level concepts with the help of concept hierarchies. For example, the city is replaced by the county.
Normalization: Normalization performed when the attribute data are scaled up o scaled down. Example: Data should fall in the range -2.0 to 2.0 post-normalization.

**Attribute construction:** these attributes are constructed and included the given set of attributes helpful for data mining.

The result of this process is a final data set that can be used in modeling.

#### Modelling
In this phase, mathematical models are used to determine data patterns.

* Based on the business objectives, suitable modeling techniques should be selected for the prepared dataset.
* Create a scenario to test check the quality and validity of the model.
* Run the model on the prepared dataset.
* Results should be assessed by all stakeholders to make sure that model can meet data mining objectives.

#### Evaluation:
In this phase, patterns identified are evaluated against the business objectives.

* Results generated by the data mining model should be evaluated against the business objectives.
* Gaining business understanding is an iterative process. In fact, while understanding, new business requirements may be raised because of data mining.
* A go or no-go decision is taken to move the model in the deployment phase.

#### Deployment:
In the deployment phase, you ship your data mining discoveries to everyday business operations.

* The knowledge or information discovered during data mining process should be made easy to understand for non-technical stakeholders.
* A detailed deployment plan, for shipping, maintenance, and monitoring of data mining discoveries is created.
* A final project report is created with lessons learned and key experiences during the project. This helps to improve the organization's business policy.

### Data Mining Techniques

#### 1.Classification:
This analysis is used to retrieve important and relevant information about data, and metadata. This data mining method helps to classify data in different classes.

#### 2. Clustering:
Clustering analysis is a data mining technique to identify data that are like each other. This process helps to understand the differences and similarities between the data.

#### 3. Regression:
Regression analysis is the data mining method of identifying and analyzing the relationship between variables. It is used to identify the likelihood of a specific variable, given the presence of other variables.

#### 4. Association Rules:
This data mining technique helps to find the association between two or more Items. It discovers a hidden pattern in the data set.

#### 5. Outer detection:
This type of data mining technique refers to observation of data items in the dataset which do not match an expected pattern or expected behavior. This technique can be used in a variety of domains, such as intrusion, detection, fraud or fault detection, etc. Outer detection is also called Outlier Analysis or Outlier mining.

#### 6. Sequential Patterns:
This data mining technique helps to discover or identify similar patterns or trends in transaction data for certain period.

#### 7. Prediction:
Prediction has used a combination of the other data mining techniques like trends, sequential patterns, clustering, classification, etc. It analyzes past events or instances in a right sequence for predicting a future event.

#### Challenges of Implementation of Data mine:
* Skilled Experts are needed to formulate the data mining queries.
* Overfitting: Due to small size training database, a model may not fit future states.
* Data mining needs large databases which sometimes are difficult to manage
* Business practices may need to be modified to determine to use the information uncovered.
* If the data set is not diverse, data mining results may not be accurate.
* Integration information needed from heterogeneous databases and global information systems could be complex

### Benefits of Data Mining:
* Data mining technique helps companies to get knowledge-based information.
* Data mining helps organizations to make the profitable adjustments in operation and production.
* The data mining is a cost-effective and efficient solution compared to other statistical data applications.
* Data mining helps with the decision-making process.
* Facilitates automated prediction of trends and behaviors as well as automated discovery of hidden patterns.
* It can be implemented in new systems as well as existing platforms
* It is the speedy process which makes it easy for the users to analyze huge amount of data in less time.
'
### Disadvantages of Data Mining
* There are chances of companies may sell useful information of their customers to other companies for money. For example, American Express has sold credit card purchases of their customers to the other companies.
* Many data mining analytics software is difficult to operate and requires advance training to work on.
* Different data mining tools work in different manners due to different algorithms employed in their design. Therefore, the selection of correct data mining tool is a very difficult task.
* The data mining techniques are not accurate, and so it can cause serious consequences in certain conditions.

# 12. Anova and Sentiment Analysis
