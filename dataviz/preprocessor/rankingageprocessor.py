from kafkanator.dataviz.creator import IndividualFairnessPreprocessorCreator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class RankingAgePreprocessor(IndividualFairnessPreprocessorCreator):
    def __init__(self):
        self.k=5
        self.algorithm = 'ball_tree'
        self.scalator = StandardScaler()


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

        print('Before NA dropping ',data.shape)
        data = data.dropna()
        print('After NA dropping ',data.shape)
        reduced = data
        for x in data.columns : 
            reduced = reduced[reduced[x] != '?']
        print('After ? dropping ',reduced.shape) 
        reduced.reset_index(inplace=True)
        reduced.sort_values(by=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'workclass_encoded', 'education_encoded', 'occupation_encoded',
       'relationship_encoded', 'race_encoded', 'sex_encoded',
       'marital-status_encoded', 'native-country_encoded'])
        print(' before dropping duplicates ', reduced.shape)
        reduced= reduced.drop_duplicates()
        print(' after dropping duplicates ', reduced.shape)
        targetret = reduced['income_encoded']
        X = reduced[['age','workclass_encoded', 'education_encoded','education-num','marital-status_encoded','occupation_encoded',  'relationship_encoded','race_encoded','sex_encoded','capital-loss','capital-gain','hours-per-week','native-country_encoded']]
        return X,targetret

    def scale(self,data):
        return self.scalator.fit_transform(data)

    def parameters(self,param):
        return {'distance' : 'minkowski','k':5,'algorithm':'ball_tree'}[param]

    def factory_method(self):
        return RankingAgePreprocessor()

    def preprocess( self,data,target_column,predictor=None ,sensitive_columns=None ):
        
        X_enc,y_enc = self.encode_columns(data,target_column)
        print ( ' after encoding , cleaning and prcpscin X shap ',X_enc.shape,' y  shape ',y_enc.shape)
        #X_enc["income"] = y_enc
        #X_enc.to_csv('readyClaude.csv')
        X_sc = self.scale(X_enc)
        hDict = {}
        nbrs = NearestNeighbors(n_neighbors=self.parameters("k"), algorithm=self.parameters("algorithm")).fit(X_sc)     
        distances, indices = nbrs.kneighbors(X_sc)

        for (i,inds) in enumerate(indices) : 
            targets = y_enc[inds[1:]]
            age = data["age"][i]
            hDict[i] = (y_enc[i], targets , age  )
        return hDict