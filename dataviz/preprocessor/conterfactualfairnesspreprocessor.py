
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kafkanator.dataviz.creator import IndividualFairnessPreprocessorCreator
from kafkanator.fairness.counterfactual import generate_counterfactual_individuals
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class ConterfactualFairnessPreprocessor(IndividualFairnessPreprocessorCreator):

    def encode_columns(self,compas_initial,target):
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
        y = compas_to_analysis['C_ScoreText']
        compas_to_analysis = compas_to_analysis.drop('C_ScoreText',axis=1)
        X = compas_to_analysis[[ 'Scale_ID',	'RawScore','DecileScore','C_Agency_Text','C_Sex_Code_Text','C_Ethnic_Code_Text','C_LegalStatus','C_Language','C_CustodyStatus','C_MaritalStatus','C_RecSupervisionLevelText','C_DisplayText','YearOfBirth']]
        print('SHAPE ', X.shape, 'top 10 index ' , X.index.values[0:20])
        return X,y

    def factory_method(self):
        return SimilarityFairnessPreprocessor()

    def sensitive_attribute_gap ( self,x, neigbors , sensitive_attributes):
        filtx = x[sensitive_attributes]
        
        filtk = neigbors.iloc[:,sensitive_attributes]
        #print ( 'x ', filtx,' neigbor ', filtk,'sa ', sensitive_attributes)  
        sumgapk = 0
        totalsumgap = []
        for k in range(0,filtk.shape[0]):
            sumgapk=0
            for s in range(0,filtk.shape[1]):
                #print ( 'k is ', k, ' s is ', s)
                d = 1 if filtk.iloc[k,s] != filtx[s] else 0 
                sumgapk = sumgapk + d
            totalsumgap.append(sumgapk)
        return sum(totalsumgap)

    def preprocess( self,data,target_column,predictor=None ,sensitive_columns=None ):
        print('predictor is preprocessing',predictor)
        X_enc,y_enc = self.encode_columns(data,target_column)
        X_enc['Target'] = y_enc
        X_enc['Ind.Type'] = ['Real']*X_enc.shape[0]
        num_cols = [i for (i,v) in enumerate(X_enc.columns.values) if v in sensitive_columns]
        print('Xenc shape ',X_enc.shape)
        X_train, X_test, y_train, y_test  = train_test_split(X_enc, target_column, test_size=0.3, random_state=42)
        X_test.reset_index(inplace=True)
        X_test = X_test.drop(columns=['index'])
        y_testr = y_test.reset_index()
        y_testr = y_testr.drop(columns=['index'])
        print('top 10 index ---- XT ' , X_test.index.values[0:20])
        hDict = {}
        returneddata = pd.DataFrame ( data = [], columns=X_enc.columns.values)
        indices_norder = []
        yencarray = y_enc.values.tolist()
        for i in range(X_test.shape[0]): 
            conterf_indivs=[]
            conterf_indivs = generate_counterfactual_individuals(X_test,i,sensitive_columns)
            print ( ' counter shape ' , conterf_indivs.shape )
            if predictor != None:
                Y = predictor.predict ( conterf_indivs.to_numpy() )
                conterf_indivs['Target'] = Y
                conterf_indivs['Ind.Type'] = 'Ctfctl'
            else:
                print ( 'you must specify a predictor inheriting transform from scikitlearn')
            if sensitive_columns != None:
                g = self.sensitive_attribute_gap ( X_test.iloc[i,:], conterf_indivs,  num_cols )
            hDict[i] = (y_enc[i], Y , g )
            X_enc = pd.concat( [X_enc,conterf_indivs],ignore_index=True )
            print ( ' XNC SHAPE ', X_enc.shape , ' appending ', X_enc.index.values[-3:])
            line = np.array([i])
            appendline = np.append(line,X_enc.index.values[-3:])
            indices_norder.append (list(appendline))
        print ( ' TOP 10 hdict ',list(hDict.items())[0:10], 'neigbors shape ' , indices_norder[0:10])
        return (hDict,indices_norder, X_enc)
        # (hDict ,indices_norder,returneddata)

    