import gower
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
from ..creator import IndividualFairnessPreprocessorCreator
from sklearn.preprocessing import StandardScaler

class SimilarityFairnessPreprocessor(IndividualFairnessPreprocessorCreator):

    def encode_columns():
        compas_filtered_columns = compas_initial.drop(['AssessmentType','Screening_Date','AssessmentID','Case_ID','Person_ID','LastName','FirstName','MiddleName','ScaleSet_ID','ScaleSet','AssessmentReason','RecSupervisionLevel','IsCompleted','IsDeleted'],axis=1)
        le = LabelEncoder()
        def reduce_to2(x):
            if x != 'Caucasian':
                return 'No Caucasian'
            else:
                return x
        compas_filtered_columns['C_Agency_Text'] = le.fit_transform( compas_filtered_columns['Agency_Text'] )
        compas_filtered_columns['C_Sex_Code_Text'] = le.fit_transform( compas_filtered_columns['Sex_Code_Text'] )
        compas_filtered_columns['C_Ethnic_Code_Text'] = compas_filtered_columns['Ethnic_Code_Text'].apply(lambda x : reduce_to2(x))
        compas_filtered_columns['C_Ethnic_Code_Text'] = le.fit_transform( compas_filtered_columns['C_Ethnic_Code_Text'] )
        compas_filtered_columns['C_LegalStatus'] = le.fit_transform( compas_filtered_columns['LegalStatus'] )
        compas_filtered_columns['C_Language'] = le.fit_transform( compas_filtered_columns['Language'] )
        compas_filtered_columns['C_CustodyStatus'] = le.fit_transform( compas_filtered_columns['CustodyStatus'] )
        compas_filtered_columns['C_MaritalStatus'] = le.fit_transform( compas_filtered_columns['MaritalStatus'] )
        compas_filtered_columns['C_RecSupervisionLevelText'] = le.fit_transform( compas_filtered_columns['RecSupervisionLevelText'] )
        compas_filtered_columns['C_DisplayText'] = le.fit_transform( compas_filtered_columns['DisplayText'] )
        compas_filtered_columns['C_ScoreText'] = le.fit_transform( compas_filtered_columns['ScoreText'] )
        compas_filtered_columns['YearOfBirth'] = compas_filtered_columns['DateOfBirth'].apply(lambda x : x.split('/')[-1])
        compas_to_analysis = compas_filtered_columns.drop(['Agency_Text', 'Sex_Code_Text' ,'Ethnic_Code_Text','LegalStatus', 'Language','CustodyStatus' ,'MaritalStatus' ,'RecSupervisionLevelText' ,'DisplayText', 'ScoreText','DateOfBirth'],axis=1)
        compas_to_analysis['YearOfBirth'] = compas_to_analysis['YearOfBirth'].astype('int64')
        tosim = compas_to_analysis[['Scale_ID',	'RawScore'	,'DecileScore'	,'C_Agency_Text',	'C_Sex_Code_Text',	'C_Ethnic_Code_Text',	'C_LegalStatus'	,'C_Language',	'C_CustodyStatus',	'C_MaritalStatus',	'C_RecSupervisionLevelText',	'C_DisplayText',	'YearOfBirth','C_ScoreText']]
        print(' before dropping duplicates ', tosim.shape)
        tosim= tosim.drop_duplicates()
        print(' after dropping duplicates ', tosim.shape)
        tosim.reset_index(inplace=True)
        tosim = tosim.drop('index',axis=1)
        y = tosim['C_ScoreText']
        tosim = tosim.drop('C_ScoreText',axis=1)
        return tosim

    def scale(self,data):
        return self.scalator.fit_transform(data)
        
    def parameters(self,param):
        return {'distance' : 'minkowski','k':5,'algorithm':'ball_tree'}[param]

    def target_distance(self,i,j):
        return abs(i-j)

    def factory_method(self):
        return SimilarityFairnessPreprocessor()

    def preprocess( self,data,target_column,predictor=None ,sensitive_columns=None ):
        hDict = {}
        tvalues = target_column.values
        nbrs = NearestNeighbors(n_neighbors=self.parameters("k"), algorithm=self.parameters("algorithm"),metric=self.parameters("distance")).fit(data)     
        distances, indices = nbrs.kneighbors(data)
        for (i,inds) in enumerate(indices) : 
            individual_neigbords = inds[1:] 
            neigb_targets = tvalues[  inds[1:] ]
            cleanedDist = distances[i,1:]
            #print ( ' PRE for individual ', i, 'itarget ',target_column.iloc[i],'neigbors ', individual_neigbords , ' targets ', neigb_targets, ' distances ', cleanedDist)
            target_distances = [self.target_distance( tvalues[i],    tvalues[j]   ) for j in individual_neigbords]
            #print ( ' POS for individual i ', 'itarget ',target_column.iloc[i],'neigbors ', individual_neigbords , ' targets ', neigb_targets, ' distances ', cleanedDist,' target  distances ', target_distances)
            if 0 in cleanedDist:
                print ( 'ZEROOOOOOOO WARNING DIVISION LOKO')
            hDict[i] = ( individual_neigbords , cleanedDist , target_distances )
        print('Outpuf of hdict ', len(list(hDict.items())))
        return hDict