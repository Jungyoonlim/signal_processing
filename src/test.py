import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/Users/jungyoonlim/EEG_emotions/data/BRMH.csv')

# Only focus on the eeg signal, so drop age, education, IQ, no, eeg.date
data.drop(["no.", "age", "eeg.date","education", "IQ"], axis=1, inplace =True)
data.drop(["sex"], axis=1,inplace= True)

# Prepare for encoding
cat_vars = ['sex', 'main.disorder','specific.disorder']
targets = ['main.disorder', 'specific.disorder']



# Drop unique columns
X = data.drop([sep_col, 'no.', 'eeg.date'], axis=1).copy(deep=True)

# columns for log transformation
logtrans_cols = X.drop(['sex', 'education', 'IQ']+targets, axis=1).columns

# Encode categorical variables (target columns and sex)
enc = OrdinalEncoder()
X[cat_vars] = enc.fit_transform(X[cat_vars])

# Save targets
md_target = X['main.disorder']
sd_target = X['specific.disorder']

# Drop targets
X.drop(['main.disorder', 'specific.disorder'], axis=1, inplace=True)

# Perform logarithmic transformation on all data except sex, education and IQ
logtrans_cols = X.drop(['sex', 'education', 'IQ'], axis=1).columns
X[logtrans_cols] = np.log(X[logtrans_cols])

# save features' names
feature_names = X.columns

# main.disorder
Xmd_train, Xmd_test, Ymd_train, Ymd_test = train_test_split(X, md_target, test_size=0.33, random_state=77)

# specific.disorder
Xsd_train, Xsd_test, Ysd_train, Ysd_test = train_test_split(X, sd_target, test_size=0.33, random_state=77)

# Scale features
# main.disorder
md_scaler = StandardScaler()
Xmd_train = md_scaler.fit_transform(Xmd_train)
Xmd_test = md_scaler.transform(Xmd_test)
# specific.disorder
sd_scaler = StandardScaler()
Xsd_train = sd_scaler.fit_transform(Xsd_train)
Xsd_test = sd_scaler.transform(Xsd_test)