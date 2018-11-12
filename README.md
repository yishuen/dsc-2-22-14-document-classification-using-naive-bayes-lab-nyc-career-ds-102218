
# Bayesian Spam Filter Lab

## Problem Statement
In this lab, we'll make use of our newfound Bayesian knowledge to classify emails as spam or not spam from the [UCI Machine Learning Repository's Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase).  

## Objectives
* Work with a real-world dataset from the UCI Machine Learning Repository.
* Classify emails as spam or not spam by making use of Naive Bayesian Classification. 
* Evaluate the quality of our classifier by building a Confusion Matrix.

Run the cell below to import everything we'll need for this lab. 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
# Do not change the seed, or else the tests will break!
np.random.seed(0)
%matplotlib inline
```

For this lab, we'll be working with the [Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase) from [UC Irvine's Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 

This dataset contains emails that have already been vectorized, as well as summary statistics about each email that can also be useful in classification.  In this case, the Data Dictionary containing the names and descriptions of each column is stored in a separate file from the dataset itself.  For ease of use, we have included the `spambase.csv` file in this repo.  However, we have not included the Data Dictionary and column names.  

In the cell below, read in the data from `spambase.csv`, store it in a DataFrame, and print the head.  

**_HINT:_** By default, pandas will automatically assume that the first row contains metadata containing the column names. Since our dataset does not have a row of metadata, pandas will mistakenly assume the values for the first email are the column names for each column.  You can prevent this by setting the `header` parameter to `None`.


```python
# Test 1: Do not change variable name!
df = pd.read_csv('spambase.csv', header=None)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.756</td>
      <td>61</td>
      <td>278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>0.28</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.94</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.132</td>
      <td>0.0</td>
      <td>0.372</td>
      <td>0.180</td>
      <td>0.048</td>
      <td>5.114</td>
      <td>101</td>
      <td>1028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.0</td>
      <td>1.23</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.12</td>
      <td>0.64</td>
      <td>0.25</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.143</td>
      <td>0.0</td>
      <td>0.276</td>
      <td>0.184</td>
      <td>0.010</td>
      <td>9.821</td>
      <td>485</td>
      <td>2259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.137</td>
      <td>0.0</td>
      <td>0.137</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.135</td>
      <td>0.0</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



As we can see, the dataset does not contain column names.  You will need to manually grab these from the [Dataset Description](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names) and create an array containing the correct column names that we can set.  

Take a minute to visit the link above and get the names of each column.  There's no python magic needed here--you'll just need to copy and paste them over in the correct order as strings in a python array.  (It's not glamorous, but it's realistic.  This is a pretty common part of the Data Science Process.)

In the cell below, create the array of column names and then set then use this array to set the correct column names for the `df` object.  

**_NOTE:_** Be sure to read the Dataset Description/Documentation carefully.  Note that the last column of the dataset (we can call it `is_spam` is the last column of the actual dataset, although the data description has it at the top, not the bottom, of the list.  Make sure you get this column name in the right place, as it will be our target variable!


```python
# Test 2: Do not change variable name!
column_names = ['word_freq_make',
    'word_freq_address',
    'word_freq_all',     
    'word_freq_3d',           
    'word_freq_our',          
    'word_freq_over',         
    'word_freq_remove',       
    'word_freq_internet',     
    'word_freq_order',        
    'word_freq_mail',         
    'word_freq_receive',      
    'word_freq_will',         
    'word_freq_people',       
    'word_freq_report',       
    'word_freq_addresses',    
    'word_freq_free',         
    'word_freq_business',     
    'word_freq_email',        
    'word_freq_you',          
    'word_freq_credit',       
    'word_freq_your',         
    'word_freq_font',         
    'word_freq_000',          
    'word_freq_money',        
    'word_freq_hp',           
    'word_freq_hpl',          
    'word_freq_george',       
    'word_freq_650',          
    'word_freq_lab',          
    'word_freq_labs',         
    'word_freq_telnet',       
    'word_freq_857',          
    'word_freq_data',         
    'word_freq_415',          
    'word_freq_85',           
    'word_freq_technology',   
    'word_freq_1999',         
    'word_freq_parts',        
    'word_freq_pm',           
    'word_freq_direct',       
    'word_freq_cs',           
    'word_freq_meeting',      
    'word_freq_original',     
    'word_freq_project',      
    'word_freq_re',           
    'word_freq_edu',          
    'word_freq_table',        
    'word_freq_conference',   
    'char_freq_;',            
    'char_freq_(',            
    'char_freq_[',            
    'char_freq_!',            
    'char_freq_$',            
    'char_freq_#',            
    'capital_run_length_average', 
    'capital_run_length_longest', 
    'capital_run_length_total',
     'is_spam'              
    ]   

df.columns = column_names
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_freq_make</th>
      <th>word_freq_address</th>
      <th>word_freq_all</th>
      <th>word_freq_3d</th>
      <th>word_freq_our</th>
      <th>word_freq_over</th>
      <th>word_freq_remove</th>
      <th>word_freq_internet</th>
      <th>word_freq_order</th>
      <th>word_freq_mail</th>
      <th>...</th>
      <th>char_freq_;</th>
      <th>char_freq_(</th>
      <th>char_freq_[</th>
      <th>char_freq_!</th>
      <th>char_freq_$</th>
      <th>char_freq_#</th>
      <th>capital_run_length_average</th>
      <th>capital_run_length_longest</th>
      <th>capital_run_length_total</th>
      <th>is_spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.756</td>
      <td>61</td>
      <td>278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>0.28</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.94</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.132</td>
      <td>0.0</td>
      <td>0.372</td>
      <td>0.180</td>
      <td>0.048</td>
      <td>5.114</td>
      <td>101</td>
      <td>1028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.0</td>
      <td>1.23</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.12</td>
      <td>0.64</td>
      <td>0.25</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.143</td>
      <td>0.0</td>
      <td>0.276</td>
      <td>0.184</td>
      <td>0.010</td>
      <td>9.821</td>
      <td>485</td>
      <td>2259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.137</td>
      <td>0.0</td>
      <td>0.137</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.135</td>
      <td>0.0</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



### Cleaning and Exploring the Dataset

Now, in the cell below, use what you've learned to clean and explore the dataset.  Make sure you check for null values, and examine the descriptive statistics for the dataset.  

Try to create at least 1 visualization during this Exploratory Data Analysis (EDA) process. 

Use the cells below for this step. 

**_Remember_**, if you need to add more cells, you can always highlight a cell, press `esc` to enter command mode, and then press `a` to add a cell above the highlighted cell, or `b` to add a cell below the highlighted cell. 


```python
# Check for Null Values
df.isna().sum().unique() # array([0], dtype=int64)  <-- 0 means that no columns contain any NaNs.  The dtype corresponds to 
                         #  the values this function returns, not the actual data types contained in the dataset

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word_freq_make</th>
      <th>word_freq_address</th>
      <th>word_freq_all</th>
      <th>word_freq_3d</th>
      <th>word_freq_our</th>
      <th>word_freq_over</th>
      <th>word_freq_remove</th>
      <th>word_freq_internet</th>
      <th>word_freq_order</th>
      <th>word_freq_mail</th>
      <th>...</th>
      <th>char_freq_;</th>
      <th>char_freq_(</th>
      <th>char_freq_[</th>
      <th>char_freq_!</th>
      <th>char_freq_$</th>
      <th>char_freq_#</th>
      <th>capital_run_length_average</th>
      <th>capital_run_length_longest</th>
      <th>capital_run_length_total</th>
      <th>is_spam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>...</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
      <td>4601.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.104553</td>
      <td>0.213015</td>
      <td>0.280656</td>
      <td>0.065425</td>
      <td>0.312223</td>
      <td>0.095901</td>
      <td>0.114208</td>
      <td>0.105295</td>
      <td>0.090067</td>
      <td>0.239413</td>
      <td>...</td>
      <td>0.038575</td>
      <td>0.139030</td>
      <td>0.016976</td>
      <td>0.269071</td>
      <td>0.075811</td>
      <td>0.044238</td>
      <td>5.191515</td>
      <td>52.172789</td>
      <td>283.289285</td>
      <td>0.394045</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.305358</td>
      <td>1.290575</td>
      <td>0.504143</td>
      <td>1.395151</td>
      <td>0.672513</td>
      <td>0.273824</td>
      <td>0.391441</td>
      <td>0.401071</td>
      <td>0.278616</td>
      <td>0.644755</td>
      <td>...</td>
      <td>0.243471</td>
      <td>0.270355</td>
      <td>0.109394</td>
      <td>0.815672</td>
      <td>0.245882</td>
      <td>0.429342</td>
      <td>31.729449</td>
      <td>194.891310</td>
      <td>606.347851</td>
      <td>0.488698</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.588000</td>
      <td>6.000000</td>
      <td>35.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.065000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.276000</td>
      <td>15.000000</td>
      <td>95.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.380000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.160000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.188000</td>
      <td>0.000000</td>
      <td>0.315000</td>
      <td>0.052000</td>
      <td>0.000000</td>
      <td>3.706000</td>
      <td>43.000000</td>
      <td>266.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.540000</td>
      <td>14.280000</td>
      <td>5.100000</td>
      <td>42.810000</td>
      <td>10.000000</td>
      <td>5.880000</td>
      <td>7.270000</td>
      <td>11.110000</td>
      <td>5.260000</td>
      <td>18.180000</td>
      <td>...</td>
      <td>4.385000</td>
      <td>9.752000</td>
      <td>4.081000</td>
      <td>32.478000</td>
      <td>6.003000</td>
      <td>19.829000</td>
      <td>1102.500000</td>
      <td>9989.000000</td>
      <td>15841.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 58 columns</p>
</div>




```python
df['is_spam'].sum() / len(df['is_spam'])
```




    0.39404477287546186




```python
# Example Visualization-- Correlation Heatmap

sns.set(style='white')

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(22,18))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0, 
            square=True, linewidths=.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1cc83f12208>




![png](output_9_1.png)


### Analysis of Exploration

Did you notice anything interesting during your EDA? Briefly explain your approach and your findings below this line:
________________________________________________________________________________________________________________________________

Student should notice that the DataFrame contains no missing values.  They may notice that many cells have values of 0.0, which could be interpreted as missing values in some contexts.  However, for our purposes, this 0.0 contains information relevant to our problem (an indicator of a real email may be when it _doesn't_ contain certain words), which means that they are not a missing value in this context. Student should have a passable rationale for the type of visualization they created, and what sort of information that visualization can provide them on the dataset.  For the case of the heatmap in this example, it shows that there is some small multicollineary present in our dataset--however, it is likely not enough to present a problem (highly correlated predictors are problematic with Naive Bayes because of the "Naive" assumption of feature independence).  The bottom row of the heatmap also shows the correlation for each predictor with our target variable, `is_spam`. 


### Creating Training and Testing Sets

Since we are using Naive Bayes for classification, we'll need to treat this like any other machine learning problem and create separate **_training sets_** and **_testing sets_** for **_holdout validation_**.  Otherwise, if we just trust the classifier's performance on the training set, we won't know for sure if the classifier has learned to detect spam emails in the real world, or just from this particular dataset.  

In the cell below:

* Store the target column in a separate variable and then remove it from the dataset. 
* Create training and testing sets using the [appropriate method](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from `sklearn`.  

**_HINT:_** We want to make sure that the training and testing samples get same distribution of spam/not spam emails.  Otherwise, our model may get a training set that doesn't contain enough of one class to learn how to tell it apart from the other.  In order to deal with this problem, we can pass in the variable containing our labels to the `stratify` parameter.  For more information, see the documentation in the link above.  


```python
# Test 3: Do not change variable names!
target = df['is_spam']
clean_df = df.drop('is_spam', axis=1, inplace=False)

# Test 4: Do not change variable names!
X_train, X_test, y_train, y_test = train_test_split(clean_df, target, stratify=target)
```

### Fitting our Classifier

Now that we have split our data into appropriate sets, we need to fit our classifier before we can make predictions and check our model's performance.

Recall what you learned about the 3 different types of Naive Bayesian Classifiers provided by `sklearn`.  Given the distribution of our data, explain why each of the following classifier types is or isn't appropriate for this problem.

**_GaussianNB:_** This one is most appropriate.  The values are continuous, and are likely generally somewhat normally distributed.  

**_BernoulliNB:_** This model is inappropriate for this dataset, since the features are continuous and real-valued, not multivariate bernoulli.  

**_MultinomialNB:_** This model is also inappropriate for this dataset, since the features are not categorical.  GaussianNB would be a better choice.  

In the cell below, create the appropriate classifier type and then `fit()` it to the appropriate training data/labels.


```python
clf = GaussianNB()
clf.fit(X_train, y_train)
```




    GaussianNB(priors=None)



### Making Predictions

Now that we have a fitted model, we can make predictions on our testing data.  

In the cell below, use the appropriate method to make predictions on the data contained inside `X_test`.


```python
preds = clf.predict(X_test)
```

### Checking Model Performance

Now that we have predictions, we can check the accuracy of our model's performance.  In order to do this, we'll use two different metrics: `accuracy_score` and `f1_score`.  For classification, accuracy is defined as the number of correct predictions (**_True Positives_** and **_True Negatives_**) divided by the total number of predictions.  

**_F1 Score_** is the harmonic mean of precision and recall. This tells us the accuracy, but penalizes the classifier heavily if it favors either **_Precision_** (Spam emails correctly identified, divided by all emails predicted to be spam) or **_Recall_** (the percentage of spam emails successfully caught, out of all spam emails) too much.  Don't worry if you aren't yet familiar with these terms--we'll cover these concepts in depth in later lessons!

In the cell below, use the appropriate helper functions from sklearn to get the accuracy and f1 scores for our model.  


```python
# Test 5: Do not change variable name!
accuracy = accuracy_score(y_test, preds)

# Test 6: Do not change variable name!
f1 = f1_score(y_test, preds)

print("Accuracy Score for model: {:.4}%".format(accuracy * 100))
print("F1 Score for model: {:.4}%".format(f1 * 100))
```

    Accuracy Score for model: 83.93%
    F1 Score for model: 82.53%
    

### Digging Deeper: Using a Confusion Matrix

Our model does pretty well, with ~81% accuracy.  However, we don't know _how_ it's failing on the 19% it got wrong.  In order to figure this out, we'll build a **_Confusion Matrix_**.

For every prediction our model makes, there are four possible outcomes:

**_True Positive:_** Our model predicted that the email was spam, and it was actually spam. 

**_True Negative:_** Our model predicted that the email was not spam, and it wasn't spam. 

**_False Positive:_** Our model predicted that the email was spam, but it wasn't.

**_False Negative:_** Our model predicted that the email wasn't spam, but it was.  


#### Question:

Which type of misclassification is preferable to the other--False Positives or False Negatives?  In this given problem, which one is preferable to the other? Explain your answer below this line:
________________________________________________________________________________________________________________________________

Overall, this is no clear preference for one over the other.  This depends entirely on the problem domain and your goals.  In this case with our spam filter, False Negatives are highly preferable to False Positives.  If a spam email accidentally sneaks through the filter, that's a small annoyance for the user.  However, if regular emails never reach the user because they are mistakenly classified as spam, that's a huge problem. 


#### Building our Confusion Matrix

In the cell below, complete the `confusion_matrix` function.  This function should take in two parameters, `predictions` and `labels`, and return a dictionary counts for `'TP', 'TN', 'FP',` and `'FN'` (True Positive, True Negative, False Positive, and False Negative, respectively).  

Once you have completed this function, use it to create Confusion Matrices for both the training and testing sets, and complete the tables in the following markdown cell.

**_HINT:_** Your labels are currently stored in a pandas series.  To make things easier, consider converting this series to a regular old python list!


```python
# Test 7: Do not change variable name!
def confusion_matrix(predictions, labels):
    labels = list(labels)
    cm = {'TP': 0, 'TN': 0, 'FP':0, 'FN':0}
    for i in range(len(predictions)):
        pred = predictions[i]
        label = labels[i]
        if pred == label:
            if pred == 1:
                cm['TP'] += 1
            else:
                cm['TN'] += 1
        else:
            if pred == 1:
                cm['FP'] += 1
            else:
                cm['FN'] += 1
    
    return cm

# Test 8: Do not change variable name!
training_preds = clf.predict(X_train)

# Test 9: Do not change variable name!
training_cm = confusion_matrix(training_preds, y_train)

# Test 10: Do not change variable name!
testing_cm = confusion_matrix(preds, y_test)

print("Training Confusion Matrix: {}".format(training_cm))
print("Testing Confusion Matrix: {}".format(testing_cm))
```

    Training Confusion Matrix: {'TP': 1295, 'TN': 1536, 'FP': 555, 'FN': 64}
    Testing Confusion Matrix: {'TP': 437, 'TN': 529, 'FP': 168, 'FN': 17}
    

### Intepreting Our Results

Complete the tables below, and then use them to answer the following questions.


|  Training Results  | **Is Spam** | **Is Email** |
|:---------------:|:-------:|:--------:|
|  **Predicted Spam** |    1295     |    555      |
| **Predicted Email** |     64    |     1536    |
<br>

|  Testing Results  | **Is Spam** | **Is Email** |
|:---------------:|:-------:|:--------:|
|  **Predicted Spam** |     437    |    168      |
| **Predicted Email** |     17    |     529     |


How many emails are getting caught up in the spam filter? How many spam emails are getting through the filter?  Is this a model you would recommend shipping to production? Why or why not?
________________________________________________________________________________________________________________________________

The False Positives rate (Predicted Spam but Is Email) are the emails getting caught in the spam filter.  The False Negatives (Predicted Email but Is Spam) are the number of spam emails slipping past the filter.  Currently, this model is weighted towards False Positives rather than False Negatives.  This means that customers wouldn't get many spam emails, but they would also have a high chance of missing out on important emails that mistakenly get caught in the spam filter.  I would not recommend shipping this model to production without tuning it to prefer False Negatives over False Positives--better for spam to get through than real emails get lost!


Don't worry about tuning the model for now--that's a lengthy process, and we'll cover it in depth in later labs.  For now, congratulations--you just built a working spam filter using Naive Bayesian Classification!


### Conclusion

In this lab, we:
* Worked with a real-world dataset from the UCI Machine Learning Repository.
* Classified emails as spam or not spam by with a Naive Bayesian Classifier. 
* Built a Confusion Matrix to evaluate the performance of our classifier.   

