# MBA-617-Business-Project
Business analysis using predictive analytics on a synthetic e-commerce data set. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scipy
!pip install scikit-learn
!pip install gdown
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: pandas in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (2.1.4)
Requirement already satisfied: numpy<2,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: numpy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: matplotlib in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.4.4)
Requirement already satisfied: numpy<2,>=1.21 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (2.8.2)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: seaborn in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (0.12.2)
Requirement already satisfied: numpy!=1.24.0,>=1.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (1.26.4)
Requirement already satisfied: pandas>=0.25 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (2.1.4)
Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scipy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.12.0)
Requirement already satisfied: numpy<1.29.0,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scipy) (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scikit-learn in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (1.7.0)
Requirement already satisfied: numpy>=1.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.8.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.12.0)
Requirement already satisfied: joblib>=1.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.2.0)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: gdown in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (5.2.0)
Requirement already satisfied: beautifulsoup4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.12.2)
Requirement already satisfied: filelock in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (3.13.1)
Requirement already satisfied: requests[socks] in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (2.31.0)
Requirement already satisfied: tqdm in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.65.0)
Requirement already satisfied: soupsieve>1.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from beautifulsoup4->gdown) (2.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2024.2.2)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (1.7.1)
import gdown



file_id='1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg'
url = f'https://drive.google.com/uc?id={file_id}'
output='ecommerce_returns_synthetic_data.csv'

gdown.download(url, output, quiet=False)
Downloading...
From: https://drive.google.com/uc?id=1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg
To: /var/www/filebrowser/.projects/ac5346fa-873d-4a66-9760-5dbaac0b2f07/ecommerce_returns_synthetic_data.csv
100%|██████████| 1.33M/1.33M [00:00<00:00, 13.6MB/s]
'ecommerce_returns_synthetic_data.csv'
df=pd.read_csv('ecommerce_returns_synthetic_data.csv', encoding='latin1')
df.shape
(10000, 17)
df.head()
Order_ID	Product_ID	User_ID	Order_Date	Return_Date	Product_Category	Product_Price	Order_Quantity	Return_Reason	Return_Status	Days_to_Return	User_Age	User_Gender	User_Location	Payment_Method	Shipping_Method	Discount_Applied
0	ORD00000000	PROD00000000	USER00000000	2023-08-05	2024-08-26	Clothing	411.59	3	Changed mind	Returned	387.0	58	Male	City54	Debit Card	Next-Day	45.27
1	ORD00000001	PROD00000001	USER00000001	2023-10-09	2023-11-09	Books	288.88	3	Wrong item	Returned	31.0	68	Female	City85	Credit Card	Express	47.79
2	ORD00000002	PROD00000002	USER00000002	2023-05-06	NaN	Toys	390.03	5	NaN	Not Returned	NaN	22	Female	City30	Debit Card	Next-Day	26.64
3	ORD00000003	PROD00000003	USER00000003	2024-08-29	NaN	Toys	401.09	3	NaN	Not Returned	NaN	40	Male	City95	PayPal	Next-Day	15.37
4	ORD00000004	PROD00000004	USER00000004	2023-01-16	NaN	Books	110.09	4	NaN	Not Returned	NaN	34	Female	City80	Gift Card	Standard	16.37
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 17 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   Order_ID          10000 non-null  object 
 1   Product_ID        10000 non-null  object 
 2   User_ID           10000 non-null  object 
 3   Order_Date        10000 non-null  object 
 4   Return_Date       5052 non-null   object 
 5   Product_Category  10000 non-null  object 
 6   Product_Price     10000 non-null  float64
 7   Order_Quantity    10000 non-null  int64  
 8   Return_Reason     5052 non-null   object 
 9   Return_Status     10000 non-null  object 
 10  Days_to_Return    5052 non-null   float64
 11  User_Age          10000 non-null  int64  
 12  User_Gender       10000 non-null  object 
 13  User_Location     10000 non-null  object 
 14  Payment_Method    10000 non-null  object 
 15  Shipping_Method   10000 non-null  object 
 16  Discount_Applied  10000 non-null  float64
dtypes: float64(3), int64(2), object(12)
memory usage: 1.3+ MB
df.nunique()
Order_ID            10000
Product_ID          10000
User_ID             10000
Order_Date            731
Return_Date           729
Product_Category        5
Product_Price        9074
Order_Quantity          5
Return_Reason           4
Return_Status           2
Days_to_Return       1249
User_Age               53
User_Gender             2
User_Location         100
Payment_Method          4
Shipping_Method         3
Discount_Applied     4317
dtype: int64
df.isna().sum()
Order_ID               0
Product_ID             0
User_ID                0
Order_Date             0
Return_Date         4948
Product_Category       0
Product_Price          0
Order_Quantity         0
Return_Reason       4948
Return_Status          0
Days_to_Return      4948
User_Age               0
User_Gender            0
User_Location          0
Payment_Method         0
Shipping_Method        0
Discount_Applied       0
dtype: int64
df.duplicated().sum()
0
# Encoding gender (simple binary example)
df['Gender_Encoded'] = df['User_Gender'].map({'Male': 0, 'Female': 1})
df = df.drop('User_Gender', axis=1)

df_filtered = df[(df['Days_to_Return'].isna()) | (df['Days_to_Return'] >= 0)].copy()
df = df[(df['Days_to_Return'].isna()) | (df['Days_to_Return'] >= 0)].copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Product_Category'])
df = df.drop('Product_Category', axis=1)

df['Return_Encoded'] = df['Return_Status'].map({'Returned': 0, 'Not Returned': 1})
df = df.drop('Return_Status', axis=1)

df['Reason_Encoded'] = le.fit_transform(df['Return_Reason'])
df = df.drop('Return_Reason', axis=1)

df['Location_Encoded'] = le.fit_transform(df['User_Location'])
df = df.drop('User_Location', axis=1)

df['Payment_Encoded'] = le.fit_transform(df['Payment_Method'])
df = df.drop('Payment_Method', axis=1)

df['Shipping_Encoded'] = le.fit_transform(df['Shipping_Method'])
df = df.drop('Shipping_Method', axis=1)
df.head(20)
Order_ID	Product_ID	User_ID	Order_Date	Return_Date	Product_Price	Order_Quantity	Days_to_Return	User_Age	Discount_Applied	Gender_Encoded	Category_Encoded	Return_Encoded	Reason_Encoded	Location_Encoded	Payment_Encoded	Shipping_Encoded
0	ORD00000000	PROD00000000	USER00000000	2023-08-05	2024-08-26	411.59	3	387.0	58	45.27	0	1	0	0	50	1	1
1	ORD00000001	PROD00000001	USER00000001	2023-10-09	2023-11-09	288.88	3	31.0	68	47.79	1	0	0	3	84	0	0
2	ORD00000002	PROD00000002	USER00000002	2023-05-06	NaN	390.03	5	NaN	22	26.64	1	4	1	4	24	1	1
3	ORD00000003	PROD00000003	USER00000003	2024-08-29	NaN	401.09	3	NaN	40	15.37	0	4	1	4	95	3	1
4	ORD00000004	PROD00000004	USER00000004	2023-01-16	NaN	110.09	4	NaN	34	16.37	1	0	1	4	79	2	2
5	ORD00000005	PROD00000005	USER00000005	2024-02-14	2024-09-22	252.12	1	221.0	46	47.61	1	2	0	1	42	1	1
6	ORD00000006	PROD00000006	USER00000006	2023-05-29	2023-08-03	382.89	2	66.0	25	28.49	0	1	0	3	46	0	0
7	ORD00000007	PROD00000007	USER00000007	2023-02-09	2024-08-01	306.39	3	539.0	67	38.91	1	2	0	2	33	0	0
9	ORD00000009	PROD00000009	USER00000009	2023-03-10	2024-01-21	294.94	3	317.0	44	45.21	0	1	0	2	81	0	2
10	ORD00000010	PROD00000010	USER00000010	2024-05-26	NaN	119.00	2	NaN	70	8.72	1	4	1	4	36	0	1
11	ORD00000011	PROD00000011	USER00000011	2024-02-08	NaN	480.48	4	NaN	54	0.27	0	3	1	4	15	2	1
12	ORD00000012	PROD00000012	USER00000012	2024-06-07	NaN	216.41	3	NaN	66	30.26	0	4	1	4	9	3	2
13	ORD00000013	PROD00000013	USER00000013	2023-07-22	NaN	198.78	2	NaN	51	36.58	0	2	1	4	84	3	1
14	ORD00000014	PROD00000014	USER00000014	2024-09-01	NaN	146.37	4	NaN	45	28.23	0	2	1	4	89	3	1
16	ORD00000016	PROD00000016	USER00000016	2024-05-09	NaN	311.08	1	NaN	52	6.66	1	1	1	4	75	3	2
17	ORD00000017	PROD00000017	USER00000017	2023-02-03	2023-06-03	400.12	1	120.0	33	18.25	0	2	0	1	27	3	2
20	ORD00000020	PROD00000020	USER00000020	2023-07-03	NaN	6.14	1	NaN	31	47.47	1	1	1	4	68	0	1
24	ORD00000024	PROD00000024	USER00000024	2024-07-27	NaN	274.29	2	NaN	18	2.49	0	4	1	4	83	2	0
25	ORD00000025	PROD00000025	USER00000025	2024-11-22	NaN	371.15	5	NaN	58	18.68	1	2	1	4	82	2	2
26	ORD00000026	PROD00000026	USER00000026	2023-03-05	NaN	232.97	5	NaN	21	9.85	1	4	1	4	98	1	0
#0 means returned 
#1 means kept
#encoded product categories
#encoded gender
print(X_train.dtypes)
Product_Price           float64
Order_Quantity            int64
User_Age                  int64
Discount_Applied        float64
Gender_Encoded            int64
Category_Encoded          int64
Reason_Encoded            int64
Location_Encoded          int64
Payment_Encoded           int64
Shipping_Encoded          int64
Order_Date_year           int32
Order_Date_month          int32
Order_Date_day            int32
Order_Date_dayofweek      int32
dtype: object
def preprocess(df):
    df = df.copy()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Order_Date_day'] = df['Order_Date'].dt.day
    df['Order_Date_dayofweek'] = df['Order_Date'].dt.dayofweek
    df['Order_Date_month'] = df['Order_Date'].dt.month
    df['Order_Date_year'] = df['Order_Date'].dt.year
    df.drop('Order_Date', axis=1, inplace=True)
    return df

df_processed = preprocess(df)
X = df_processed.drop(['Return_Encoded','Order_ID','Product_ID','User_ID','Return_Date','Days_to_Return','Reason_Encoded'], axis=1)
y = df_processed['Return_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Assuming X_train contains date columns
# Step 1: Identify date columns
date_columns = X_train.select_dtypes(include=['object']).columns

# Step 2: Convert date strings to datetime objects and extract useful features
for col in date_columns:
    # Check if column contains date strings
    try:
        X_train[col] = pd.to_datetime(X_train[col])
        # Extract useful features from dates
        X_train[f'{col}_year'] = X_train[col].dt.year
        X_train[f'{col}_month'] = X_train[col].dt.month
        X_train[f'{col}_day'] = X_train[col].dt.day
        X_train[f'{col}_dayofweek'] = X_train[col].dt.dayofweek
        # Drop the original date column
        X_train = X_train.drop(col, axis=1)
    except:
        # If conversion fails, it might not be a date column
        pass

# Step 3: Handle any remaining categorical features
categorical_columns = X_train.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X_train[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, index=X_train.index)
    X_train = X_train.drop(categorical_columns, axis=1)
    X_train = pd.concat([X_train, encoded_df], axis=1)

# Now train the model with the processed data
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

RandomForestClassifier
?i
Parameters
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,rfc_pred))
              precision    recall  f1-score   support

           0       0.48      0.25      0.32       771
           1       0.69      0.86      0.76      1476

    accuracy                           0.65      2247
   macro avg       0.58      0.55      0.54      2247
weighted avg       0.61      0.65      0.61      2247

print(confusion_matrix(y_test,rfc_pred))
[[ 190  581]
 [ 209 1267]]
cm_rfc=confusion_matrix(y_test,rfc_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix RFC')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
No description has been provided for this image
from sklearn.ensemble import RandomForestClassifier
rfc_importance = pd.Series(rfc.feature_importances_, index=X_train.columns)
import matplotlib.pyplot as plt

# Sort and plot Random Forest feature importance
rfc_importance.sort_values(ascending=False).plot(
    kind="bar", color="orange", figsize=(10, 5)
)

plt.title("Random Forest Feature Importance")
plt.ylabel("Importance Score")
plt.xlabel("Feature Names")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
%matplotlib inline
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scipy
!pip install scikit-learn
!pip install gdown
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: pandas in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (2.1.4)
Requirement already satisfied: numpy<2,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: numpy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: matplotlib in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.4.4)
Requirement already satisfied: numpy<2,>=1.21 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (2.8.2)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: seaborn in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (0.12.2)
Requirement already satisfied: numpy!=1.24.0,>=1.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (1.26.4)
Requirement already satisfied: pandas>=0.25 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (2.1.4)
Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scipy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.12.0)
Requirement already satisfied: numpy<1.29.0,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scipy) (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scikit-learn in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (1.7.0)
Requirement already satisfied: numpy>=1.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.8.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.12.0)
Requirement already satisfied: joblib>=1.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.2.0)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: gdown in /home/e280e098-9446-4486-8caf-5799168391b0/.local/lib/python3.10/site-packages (5.2.0)
Requirement already satisfied: beautifulsoup4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.12.2)
Requirement already satisfied: filelock in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (3.13.1)
Requirement already satisfied: requests[socks] in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (2.31.0)
Requirement already satisfied: tqdm in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.65.0)
Requirement already satisfied: soupsieve>1.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from beautifulsoup4->gdown) (2.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2024.2.2)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (1.7.1)
import gdown



file_id='1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg'
url = f'https://drive.google.com/uc?id={file_id}'
output='ecommerce_returns_synthetic_data.csv'

gdown.download(url, output, quiet=False)
Downloading...
From: https://drive.google.com/uc?id=1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg
To: /var/www/filebrowser/.projects/ac5346fa-873d-4a66-9760-5dbaac0b2f07/ecommerce_returns_synthetic_data.csv
100%|██████████| 1.33M/1.33M [00:00<00:00, 27.8MB/s]
'ecommerce_returns_synthetic_data.csv'
df=pd.read_csv('ecommerce_returns_synthetic_data.csv', encoding='latin1')
df.head()
Order_ID	Product_ID	User_ID	Order_Date	Return_Date	Product_Category	Product_Price	Order_Quantity	Return_Reason	Return_Status	Days_to_Return	User_Age	User_Gender	User_Location	Payment_Method	Shipping_Method	Discount_Applied
0	ORD00000000	PROD00000000	USER00000000	2023-08-05	2024-08-26	Clothing	411.59	3	Changed mind	Returned	387.0	58	Male	City54	Debit Card	Next-Day	45.27
1	ORD00000001	PROD00000001	USER00000001	2023-10-09	2023-11-09	Books	288.88	3	Wrong item	Returned	31.0	68	Female	City85	Credit Card	Express	47.79
2	ORD00000002	PROD00000002	USER00000002	2023-05-06	NaN	Toys	390.03	5	NaN	Not Returned	NaN	22	Female	City30	Debit Card	Next-Day	26.64
3	ORD00000003	PROD00000003	USER00000003	2024-08-29	NaN	Toys	401.09	3	NaN	Not Returned	NaN	40	Male	City95	PayPal	Next-Day	15.37
4	ORD00000004	PROD00000004	USER00000004	2023-01-16	NaN	Books	110.09	4	NaN	Not Returned	NaN	34	Female	City80	Gift Card	Standard	16.37
selected_features=['Product_Category','Product_Price','Order_Quantity','Discount_Applied','Return_Status']
df_select=df[selected_features].copy()
df_select.head()
Product_Category	Product_Price	Order_Quantity	Discount_Applied	Return_Status
0	Clothing	411.59	3	45.27	Returned
1	Books	288.88	3	47.79	Returned
2	Toys	390.03	5	26.64	Not Returned
3	Toys	401.09	3	15.37	Not Returned
4	Books	110.09	4	16.37	Not Returned
import pandas as pd

# One-hot encode the Product_Category column
df_encoded = pd.get_dummies(df_select, columns=['Product_Category'], drop_first=True)
#KMeans calculates distances between points. If you label encode categories like:

#Clothing → 0

#Books → 1

#Toys → 2

#…it assumes Toys are “twice” as far from Clothing as Books are, which isn’t meaningful. One-hot avoids this trap by treating each category as orthogonal.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded['Return_Status'] = le.fit_transform(df_encoded['Return_Status'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = df_encoded.drop(columns=['Return_Status'])
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

y = df_encoded['Return_Status']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
df_select.isnull().sum()
Product_Category    0
Product_Price       0
Order_Quantity      0
Discount_Applied    0
Return_Status       0
dtype: int64
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

KNeighborsClassifier
?i
Parameters
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
[[496 513]
 [463 528]]
print(classification_report(y_test,pred))
              precision    recall  f1-score   support

           0       0.52      0.49      0.50      1009
           1       0.51      0.53      0.52       991

    accuracy                           0.51      2000
   macro avg       0.51      0.51      0.51      2000
weighted avg       0.51      0.51      0.51      2000

error_rate = []

# Will take some time
for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
Text(0, 0.5, 'Error Rate')
No description has been provided for this image
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
WITH K=1


[[496 513]
 [463 528]]


              precision    recall  f1-score   support

           0       0.52      0.49      0.50      1009
           1       0.51      0.53      0.52       991

    accuracy                           0.51      2000
   macro avg       0.51      0.51      0.51      2000
weighted avg       0.51      0.51      0.51      2000

# NOW WITH K=3
knn3 = KNeighborsClassifier(n_neighbors=3)

knn3.fit(X_train,y_train)
pred3 = knn3.predict(X_test)

print('WITH K=3')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
WITH K=3


[[499 510]
 [472 519]]


              precision    recall  f1-score   support

           0       0.51      0.49      0.50      1009
           1       0.50      0.52      0.51       991

    accuracy                           0.51      2000
   macro avg       0.51      0.51      0.51      2000
weighted avg       0.51      0.51      0.51      2000

# NOW WITH K=4
knn4 = KNeighborsClassifier(n_neighbors=4)

knn4.fit(X_train,y_train)
pred4 = knn4.predict(X_test)

print('WITH K=4')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
WITH K=4


[[499 510]
 [472 519]]


              precision    recall  f1-score   support

           0       0.51      0.49      0.50      1009
           1       0.50      0.52      0.51       991

    accuracy                           0.51      2000
   macro avg       0.51      0.51      0.51      2000
weighted avg       0.51      0.51      0.51      2000

# NOW WITH K=5
knn5 = KNeighborsClassifier(n_neighbors=5)

knn5.fit(X_train,y_train)
pred5 = knn5.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
WITH K=5


[[499 510]
 [472 519]]


              precision    recall  f1-score   support

           0       0.51      0.49      0.50      1009
           1       0.50      0.52      0.51       991

    accuracy                           0.51      2000
   macro avg       0.51      0.51      0.51      2000
weighted avg       0.51      0.51      0.51      2000

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred4)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for KNN Model')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn
%matplotlib inline
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install scipy
!pip install scikit-learn
!pip install gdown
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: pandas in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (2.1.4)
Requirement already satisfied: numpy<2,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (1.26.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: numpy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: matplotlib in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.4.4)
Requirement already satisfied: numpy<2,>=1.21 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib) (2.8.2)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: seaborn in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (0.12.2)
Requirement already satisfied: numpy!=1.24.0,>=1.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (1.26.4)
Requirement already satisfied: pandas>=0.25 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (2.1.4)
Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from seaborn) (3.8.0)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.25.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)
Requirement already satisfied: packaging>=20.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (23.2)
Requirement already satisfied: pillow>=6.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (10.2.0)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from pandas>=0.25->seaborn) (2023.3)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scipy in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (1.12.0)
Requirement already satisfied: numpy<1.29.0,>=1.22.4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scipy) (1.26.4)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: scikit-learn in ./.local/lib/python3.10/site-packages (1.7.0)
Requirement already satisfied: numpy>=1.22.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.8.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.12.0)
Requirement already satisfied: joblib>=1.2.0 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from scikit-learn) (1.2.0)
Requirement already satisfied: threadpoolctl>=3.1.0 in ./.local/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
Defaulting to user installation because normal site-packages is not writeable
Looking in links: /usr/share/pip-wheels
Requirement already satisfied: gdown in ./.local/lib/python3.10/site-packages (5.2.0)
Requirement already satisfied: beautifulsoup4 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.12.2)
Requirement already satisfied: filelock in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (3.13.1)
Requirement already satisfied: requests[socks] in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (2.31.0)
Requirement already satisfied: tqdm in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from gdown) (4.65.0)
Requirement already satisfied: soupsieve>1.2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from beautifulsoup4->gdown) (2.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (2024.2.2)
Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages (from requests[socks]->gdown) (1.7.1)
import gdown



file_id='1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg'
url = f'https://drive.google.com/uc?id={file_id}'
output='ecommerce_returns_synthetic_data.csv'

gdown.download(url, output, quiet=False)
Downloading...
From: https://drive.google.com/uc?id=1_pB6f7X-wV40HFSvPfHRktbNSqfjA2Fg
To: /home/e280e098-9446-4486-8caf-5799168391b0/ecommerce_returns_synthetic_data.csv
100%|██████████| 1.33M/1.33M [00:00<00:00, 117MB/s]
'ecommerce_returns_synthetic_data.csv'
df=pd.read_csv('ecommerce_returns_synthetic_data.csv', encoding='latin1')
df['Return_Date'] = pd.to_datetime(df['Return_Date'])
df.set_index('Return_Date', inplace=True)

# Aggregate return volume by month
monthly_returns = df.resample('M').size()  # count of returns per month

# Optional: Plot the time series
monthly_returns.plot(figsize=(10, 4), title='Monthly Return Volume')
plt.xlabel('Month')
plt.ylabel('Returns')
plt.show()
No description has been provided for this image
from statsmodels.tsa.seasonal import STL

# Apply STL decomposition
stl = STL(monthly_returns, seasonal=13)  # 13 works well for monthly seasonality
result = stl.fit()

# Plot decomposed components
result.plot()
plt.show()
No description has been provided for this image
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(monthly_returns.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
ADF Statistic: -3.499930575826343
p-value: 0.007988877200341972
ts_diff = monthly_returns.diff().dropna()
adfuller(ts_diff)
(-1.3260472372037924,
 0.6171730387044221,
 8,
 14,
 {'1%': -4.01203360058309,
  '5%': -3.1041838775510207,
  '10%': -2.6909873469387753},
 108.45471325865432)
from sklearn.model_selection import train_test_split

train = monthly_returns[:-12]
test = monthly_returns[-12:]
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Fit SARIMA model
model = SARIMAX(train,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()

# Forecast the test period
forecast = results.get_forecast(steps=12)
predicted = forecast.predicted_mean

# Evaluation metrics
mae = mean_absolute_error(test, predicted)
mse = mean_squared_error(test, predicted)
rmse = np.sqrt(mse)

# Print results
print("Forecasted values:")
print(predicted)

print(f"\nMAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =            5     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f= -0.00000D+00    |proj g|=  0.00000D+00

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
    5      0      1      0     0     0   0.000D+00  -0.000D+00
  F =  -0.0000000000000000     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
Forecasted values:
2024-01-31    209.0
2024-02-29    171.0
2024-03-31    203.0
2024-04-30    231.0
2024-05-31    238.0
2024-06-30    214.0
2024-07-31    226.0
2024-08-31    222.0
2024-09-30    208.0
2024-10-31    203.0
2024-11-30    208.0
2024-12-31    217.0
Freq: M, Name: predicted_mean, dtype: float64

MAE: 15.67
MSE: 380.00
RMSE: 19.49
/opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency M will be used.
  self._init_dates(dates, freq)
/opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages/statsmodels/tsa/base/tsa_model.py:473: ValueWarning: No frequency information was provided, so inferred frequency M will be used.
  self._init_dates(dates, freq)
/opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages/statsmodels/tsa/statespace/sarimax.py:866: UserWarning: Too few observations to estimate starting parameters for ARMA and trend. All parameters except for variances will be set to zeros.
  warn('Too few observations to estimate starting parameters%s.'
/opt/conda/envs/anaconda-ai-2024.04-py310/lib/python3.10/site-packages/statsmodels/tsa/statespace/sarimax.py:866: UserWarning: Too few observations to estimate starting parameters for seasonal ARMA. All parameters except for variances will be set to zeros.
  warn('Too few observations to estimate starting parameters%s.'
 This problem is unconstrained.
import matplotlib.pyplot as plt
import pandas as pd

# Create a Series with forecasted values and test index
forecast_series = pd.Series(predicted.values, index=test.index)

# Plot actual vs forecasted
plt.figure(figsize=(10, 5))
plt.plot(train.index[-60:], train[-60:], label='Historical (Train)', color='navy')
plt.plot(test.index, test, label='Actual (Test)', color='green')
plt.plot(test.index, forecast_series, label='SARIMA Forecast', linestyle='--', color='darkorange')

plt.title('SARIMA Forecast vs Actual Monthly Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.ylim(150, 250)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
