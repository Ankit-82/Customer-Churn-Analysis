##Customer Churn Analysis


scaler = StandardScaler()
standardized_df = scaler.fit_transform(new_df)
print(standardized_df)

import statsmodels

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

teli_data_base = pd.read_csv('/content/Churn_Modelling.csv')

univariant analysis

telco_new = teli_data_base[['Geography', 'Gender', 'Exited']]

print(telco_new )

for i , predictor in enumerate(telco_new.drop(columns=['Exited'])):
    plt.figure()
    sns.countplot(data = telco_new,x=predictor,hue = 'Exited')

Bivarient Analysis

sns.histplot(x ="Gender", hue = "Geography" , data = telco_new, stat ="count" , multiple ="dodge")

Feature encoding

df.info()

df.Gender.value_counts()

df.Gender.mode()

df['Gender'] = df['Gender'].fillna('Male')

df.info()

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['New'] = le.fit_transform(df.Gender.values)

df.head(3)

one_hot = pd.get_dummies(df['Geography'])

one_hot

df_dummies = pd.get_dummies(df)
df_dummies.head(4)

df = pd.read_csv('/content/CustomerChurn.csv')
df

df.shape

df.columns.values

df.dtypes

df.describe()

df['Churn'].value_counts()

df['Churn'].value_counts()/len(df)*100

df['Churn'].value_counts().plot(kind ='barh',figsize= (5,6) )
plt.xlabel('Count')
plt.ylabel('Table Variable')
plt.title( "Churn Table" , y =1.02)

new_df= df.copy()

new_df.info()

new_df

new_df.TotalCharges = pd.to_numeric(new_df.TotalCharges , errors = 'coerce')

new_df.isnull().sum()

new_df.info()

messing value treatment
## since % of messing values are very low

#remove messing values
new_df.dropna(inplace = True)
#or
new_df.dropna(how ='any', inplace = True)

new_df.isnull().sum()

## divide customers into bins based on tanure eg (<12 months , 1-12 months , etc

labels = ["{0} - {1}".format(i, i+11) for i in range(1,72,12)]
new_df['Tanure_group'] = pd.cut(new_df.tenure, range(1,80,12), right = False, labels = labels)

new_df['Tanure_group'].value_counts()

##step 6 : remove column which are not required for processing

new_df.drop(columns= ['customerID','tenure'], axis = 1, inplace = True)

new_df

## univariate Analysis

for i, predictor in enumerate(new_df.drop(columns = ['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data = new_df , x = predictor , hue = 'Churn')

##Numerical Analysis

new_df['gender'].value_counts()

new_df_target0 = new_df[new_df["Churn"] =='No']
new_df_target1 = new_df[new_df["Churn"] =='Yes']

new_df_target1.gender.value_counts()

pd.crosstab(new_df.PaymentMethod, new_df.Churn)

##Bivarient Anaysis

convert yes and no into 1 & 0

new_df['Churn'] = np.where(new_df.Churn =='Yes',1,0)

new_df.head()

new_df= df.copy()

new_df_dummies = pd.get_dummies(new_df)
new_df_dummies.head()

Mth = sns.kdeplot(new_df_dummies.MonthlyCharges[(new_df_dummies["Churn"] == 0)],color = "Red",fill = True)
Mth = sns.kdeplot(new_df_dummies.MonthlyCharges[(new_df_dummies["Churn"] == 1)],ax = Mth,color ="Blue",fill = True)
Mth.legend(["No Churn","Churn"],loc ='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')