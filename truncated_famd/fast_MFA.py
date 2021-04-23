# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:43:24 2019

@author: luyao.li
"""


from .fast_MCA import MCA
from .fast_PCA import PCA

import six
import numpy as np
import pandas as pd 
from sklearn.utils import check_array

from  .outils import _pearsonr
class MFA(PCA):
    def __init__(self,standard_scaler=False,n_components=2,svd_solver='auto',whiten=False,copy=True,
                 tol=None,iterated_power=2,batch_size =None,random_state=None,groups=None):
        for k,v in locals().items():
            if k!='self':
                setattr(self,k,v)

                
    def fit(self,X,y=None):
        if self.groups is None:
            raise ValueError('Must input group vars')

        check_array(X,dtype=[six.string_types[0],np.number])
        
        if not isinstance(X,(pd.DataFrame,pd.SparseDataFrame)):
            X=pd.DataFrame(X) 
        #determine which group is all_numeric
        self.partial_factor_analysis_={}
        for name, cols in  self.groups.items():
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            #all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_categorical_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
        # Run a factor analysis in each group   
       
            if all_num:
                fa = PCA(
                    standard_scaler=self.standard_scaler,
                    n_components=self.n_components,
                    copy=self.copy,
                    random_state=self.random_state,
                    svd_solver=self.svd_solver,
                    whiten=self.whiten,
                    tol=self.tol,
                    iterated_power=self.iterated_power,
                    batch_size=self.batch_size
                
                )
            else:
                fa = MCA(
                            n_components=self.n_components,
                             svd_solver=self.svd_solver,
                             copy=self.copy,
                             tol=self.tol,
                             iterated_power=self.iterated_power,
                             batch_size =self.batch_size,
                             random_state=self.random_state
                )

            self.partial_factor_analysis_[name] = fa.fit(X.loc[:, cols])  
        
        X_global=self._X_global(X)
        self._usecols= X_global.columns
        self= super().fit(X_global)
            
        return self
    
    def _X_global(self,X) :
        X_globals=[] 
        for name, cols in  self.groups.items():
            X_partial= X.loc[:,cols]
            _estimator = self.partial_factor_analysis_[name]
            if _estimator.__class__.__name__ =='PCA':
                if hasattr(_estimator,'scaler_'):
                    X_partial=  pd.DataFrame(_estimator.scaler_.transform(X_partial),columns =cols)          
            else:
                X_partial=_estimator.one_hot.transform(X_partial).loc[:,_estimator._usecols].to_dense()
            X_globals.append(X_partial/ _estimator.singular_values[0] )

        X_global=pd.concat(X_globals,axis=1 )
        return  X_global
    
    def transform(self,X,y =None):
        X_global=self._X_global(X)
        return len(X_global) ** 0.5 *super().transform(X_global)
    
    def fit_transform(self,X,y=None):
        X_global=self._X_global(X)
        self._usecols= X_global.columns
        return len(X_global) ** 0.5 *super().fit_transform(X_global)        
        
    
    def column_correlation(self,X,same_input=True):
        X_global=  self._X_global(X)
        if   same_input: #X is fitted and the the data fitting and the data transforming is the same
            X_t=len(X_global) ** 0.5 *super().transform(X_global)
        else:
            X_t=len(X_global) ** 0.5 *super().fit_transform(X_global)
        
        return pd.DataFrame({index_comp:{ 
                            col_name: _pearsonr(X_t[:,index_comp],X_global.loc[:,col_name].values.to_dense())
                              for col_name in  X_global
                                    }
                                    for index_comp  in range(X_t.shape[1])})
        
        


                
            
                
                

        
            
            
            
            
            
            
        
        
        