import xgboost as xgb



class XGBoostClassifier:
    def __init__(self,params, num_boost_round,early_stopping):
        self.data_train=''
        self.data_valid=''
        self.watch_list=''
        self._params=params
        self._num_boost_round=num_boost_round
        self._early_stopping=early_stopping

    def fit(self,X_train,y_train,X_test,y_test):
        self.data_train=xgb.DMatrix(X_train,label=y_train)
        self.data_valid=xgb.DMatrix(X_test,label=y_test)
        self.watch_list=[(self.data_valid,'evals'),(self.data_train,'train')]
        self._model=xgb.train(self._params,self.data_train,num_boost_round=self._num_boost_round,evals=self.watch_list,early_stopping_rounds=self._early_stopping)
        return self._model

    def predict(self,X_test):

        return self._model.predict(xgb.DMatrix(X_test))

    def getModel(self):
        return self._model







