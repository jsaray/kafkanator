from ..creator import IndividualFairnessPreprocessorCreator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
class ConsistencyFairnessPreprocessor(IndividualFairnessPreprocessorCreator):
    def __init__(self):
        self.k=5
        self.algorithm = 'ball_tree'
        self.scalator = StandardScaler()

    def d(x1,x2):
        return self.d(x1,x2)

    def parameters(self,param):
        return {'distance' : 'minkowski','k':5,'algorithm':'ball_tree'}[param]

    def factory_method(self):
        return ConsistencyFairnessPreprocessor()

    def encode_columns(self,data,target):
        data['race'] = data['race'].apply(lambda x : 1 if x=='White' else 0 )
        categorical_columns = ['workclass', 'education',  'occupation', 'relationship', 'race', 'sex','marital-status','native-country','income']
        data['income'] = target
        # Create an instance of LabelEncoder
        label_encoder = LabelEncoder()

        # Encode each categorical column and create new feature columns
        for col in categorical_columns:
            encoded_col_name = f"{col}_encoded"
            data[encoded_col_name] = label_encoder.fit_transform(data[col])

        #print('Before NA dropping ',data.shape)
        data = data.dropna()
        #print('After NA dropping ',data.shape)
        reduced = data
        for x in data.columns : 
            reduced = reduced[reduced[x] != '?']
        print('After ? dropping ',reduced.shape) 
        #reduced.reset_index(inplace=True)
        reduced.sort_values(by=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'workclass_encoded', 'education_encoded', 'occupation_encoded',
       'relationship_encoded', 'race_encoded', 'sex_encoded',
       'marital-status_encoded', 'native-country_encoded'])
        #print(' before dropping duplicates ', reduced.shape)
        reduced= reduced.drop_duplicates()
        #reduced.reset_index(inplace=True)
        #print(' after dropping duplicates ', reduced.shape)
        targetret = reduced['income_encoded']
        X = reduced[['age','workclass_encoded', 'education_encoded','education-num','marital-status_encoded','occupation_encoded',  'relationship_encoded','race_encoded','sex_encoded','capital-loss','capital-gain','hours-per-week','native-country_encoded']]
        return X,targetret
    
    def scale(self,data):
        return self.scalator.fit_transform(data)
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
        X_enc,y_enc = self.encode_columns(data,target_column)
        X_enc.to_csv('pcsdf.csv')
        num_cols = [i for (i,v) in enumerate(X_enc.columns.values) if v in sensitive_columns]
        #print ( ' after encoding , cleaning and prcpscin X shap ',X_enc.shape,' y  shape ',y_enc.shape, 'Xenc index ', X_enc.iloc[0:20,:],'yenc top 5',y_enc[0:30])
        X_sc = self.scale(X_enc)
        hDict = {}
        nbrs = NearestNeighbors(n_neighbors=self.parameters("k"), algorithm=self.parameters("algorithm")).fit(X_sc)     
        distances, indices_norder = nbrs.kneighbors(X_sc)
        np.savetxt("kneigb.csv", indices_norder , delimiter = ",",fmt='%f')
        #print('file printed')
        hshConv = {}
        for (i,inds_n) in enumerate(indices_norder) : 
            inds =   np.delete(inds_n, inds_n == i)
            if len(inds) < len(inds_n):
                if sensitive_columns != None:
                    g = self.sensitive_attribute_gap ( X_enc.iloc[i,:], X_enc.iloc[inds,:],num_cols )
                    #g = self.sensitive_attribute_gap ( X_enc.iloc[i,:], X_enc.iloc[inds_n[1:],:],num_cols )
                    targets = y_enc.iloc[inds]
                    hDict[i] = (y_enc.iloc[i], targets, g   )
        p=0
        for i in X_enc.index.values:
            hshConv[i] = p
            p = p+1
        return (hDict , indices_norder, hshConv)