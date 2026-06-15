import unittest
from unittest import result
import pandas as pd
from kafkanator.fairness.metrics import fairness_metrics_table
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import xgboost as xgb
import numpy as np

class GroupFairnessTest(unittest.TestCase):

    def setUp(self):
        adult = fetch_ucirepo(id=2) 
        # data (as pandas dataframes) 
        self.data = adult.data.features 
        self.y = adult.data.targets 
        # metadata 
        print(adult.metadata) 
        # variable information 
        print(adult.variables) 
        # Data cleaning
        self.y['income'] = self.y['income'].apply(lambda x: x.replace('.',''))
        self.data['sex'] = self.data['sex'].str.strip()
        self.data['race'] = self.data['race'].str.strip()
        self.data['marital-status'] = self.data['marital-status'].str.strip()
        self.data['native-country'] = self.data['native-country'].str.strip()
        self.data['race'] = self.data['race'].apply(lambda x : 1 if x=='White' else 0 )
        self.data[self.data['sex']=='Male'].shape
        self.data[self.data['sex']=='Female'].shape
        self.categorical_columns = ['workclass', 'education',  'occupation', 'relationship', 'race', 'sex','marital-status','native-country']
        
        #Data Preprocessing
        self.label_encoder = LabelEncoder()
        for col in self.categorical_columns:
            encoded_col_name = f"{col}_encoded"
            self.data[encoded_col_name] = self.label_encoder.fit_transform(self.data[col])
        Y = self.label_encoder.fit_transform(self.y['income'])
        X = self.data[['age','workclass_encoded', 'education_encoded','education-num','marital-status_encoded','occupation_encoded',	'relationship_encoded','race_encoded','sex_encoded','capital-loss','capital-gain','hours-per-week','native-country_encoded']]
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3)
        # Model building
        classifier = xgb.XGBClassifier(objective="binary:logistic")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        fairness_analysis = np.transpose(np.array([X_test['sex_encoded'].values, y_pred,y_test]))
        self.fairness_analysis_df = pd.DataFrame(data=fairness_analysis,columns=['sex_encoded','prediction','reality'])

    def test_group_fairness(self):
        sp = fairness_metrics_table(self.fairness_analysis_df,['sex_encoded'],'prediction','reality',aggregate_metrics=False)
        print(sp.shape)
        for index, row in sp.iterrows():
            self.assertLess (row['0'],1,'table must contain values between 0 and 1')
            self.assertGreater (row['0'],0,'table must contain values between 0 and 1')

            self.assertLess (row['1'],1,'table must contain values between 0 and 1')
            self.assertGreater (row['1'],0,'table must contain values between 0 and 1')


if __name__ == "__main__":
    unittest.main()
