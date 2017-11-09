import lightgbm as lgb

class LightGbmClassifier:
    def __init__(self,params,num_boost_round,early_stopping):
        self._model=''
        self.lgb_train=''
        self.lgb_test=''
        self._params = params
        self._num_boost_round = num_boost_round
        self._early_stopping = early_stopping

    def fit(self,X_train,y_train,X_test,y_test):
        self.lgb_train=lgb.Dataset(X_train,y_train)
        self.lgb_test=lgb.Dataset(X_test,y_test,reference=self.lgb_train)
        self._model=lgb.train(self._params,self.lgb_train,num_boost_round=self._num_boost_round,valid_sets=self.lgb_test,early_stopping_rounds=self._early_stopping)
        return self._model

    def predict(self,lgb_test):
        return self._model.predict(lgb_test,num_iteration=self._model.best_iteration)

    def getModel(self):
        return self._model