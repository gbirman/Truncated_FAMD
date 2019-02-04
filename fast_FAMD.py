# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:39:04 2019

@author: luyao.li
"""

from fast_MFA  import MFA
import pandas as pd
import numpy as np

class  FAMD(MFA):
    def __init__(self,n_components=2,svd_solver='auto',copy=True,
             tol=None,iterated_power=2,batch_size =None,random_state=None):
        super().__init__(
                         n_components=n_components,
                         svd_solver=svd_solver,
                         copy=copy,
                         tol=tol,
                         iterated_power=iterated_power,
                         batch_size =batch_size,
                         random_state=random_state)   
    
    def fit(self,X,y=None):
        if not isinstance(X,(pd.DataFrame,pd.SparseDataFrame)):
            X=pd.DataFrame(X) 
        _numric_columns= X.select_dtypes(include=np.number).columns
        _category_columns=X .select_dtypes(include=['object','category']).columns
        self.groups= {'Numerical':_numric_columns ,'Categorical':_category_columns }
        
        return  super().fit(X)
    
if  __name__ =='__main__':
    
#    1)
#  File "E:\1113蓝海数据建模\fast_FAMD\fast_MFA.py", line 29, in fit
#    check_array(X,dtype=[six.string_types,np.number])
#    TypeError: data type not understood:
#        six.string_types -> tuple( str,) :six.string_types[0]
#    
#    2)
#
#AttributeError: 'FAMD' object has no attribute 'n_iter':iterated_power
#3)X.select_dtypes(include=np.number).columns -> empty Int64Index:
#    X = pd.DataFrame( np.random.randint(0,1000,size=(10000,500)) ,dtype=int)
#    test_arr=pd.DataFrame( np.random.choice(list('abcd'),size=(10000,100),replace=True),dtype=str) 
#    test_X= pd.concat([X,test_arr],axis=1,ignore_index=True)
#4)   
#  File "E:\1113蓝海数据建模\fast_FAMD\outils.py", line 24, in fit
#    self.column_names_ = self.get_feature_names(X.columns if hasattr( X,'columns') else None)
#
#  File "C:\Users\admin\Anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py", line 707, in get_feature_names
#    input_features[i] + '_' + six.text_type(t) for t in cats[i]]
#
#  File "C:\Users\admin\Anaconda3\lib\site-packages\sklearn\preprocessing\_encoders.py", line 707, in <listcomp>
#    input_features[i] + '_' + six.text_type(t) for t in cats[i]]
#
#TypeError: ufunc 'add' did not contain a loop with signature matching types dtype('<U21') dtype('<U21') dtype('<U21')         
#原因在于 原始变量名为 int ,dtype('<U32')是字符串格式:
#将所有变量名转化为 str 
#
#5)
#  File "E:\1113蓝海数据建模\fast_FAMD\fast_MFA.py", line 78, in _X_global
#    X_global.append( X_partial / self.partial_factor_analysis_[name].singular_values[0]  )
#    TypeError: Could not operate 0.10558250376496033
#    with block values unsupported operand type(s) for /: 'str' and 'float'
#    X_partial 可以为str values:
#
#        if self.partial_factor_analysis_[name].__class__.__name__ =='PCA':
#                X_partial=  self.partial_factor_analysis_[name].scaler_.transform(X_partial)
#        else:
#                X_partial=self.partial_factor_analysis_[name].one_hot.transform(X_partial)
 
    
    X = pd.DataFrame( np.random.randint(0,1000,size=(10000,500)) ,dtype=int)
    test_arr=pd.DataFrame( np.random.choice(list('abcd'),size=(10000,100),replace=True),dtype=str) 
    test_X= pd.concat([X,test_arr],axis=1,ignore_index=True)
    test_X.rename(columns = lambda c:str(c),inplace=True)
    print(test_X.shape)
    famd=FAMD()
    famd.fit(test_X)
    X_t=famd.transform(test_X)
    
    assert X_t.shape[1] == famd.n_components,ValueError("")
        
                          
