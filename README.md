
#FAST_FAMD: 
##Description:
Scalable factor anlysis for mixed and sparse data


##Subclasses:
###MCA:Multiply Category Aalysis class
###PCA:Principal Component Analysis class
###MFA:Multiply Factor Analysis class



##Example:
    from fast_FAMD import FAMD
	famd=FAMD(n_components=2,
			svd_solver='auto',copy=True,
         	tol=None,iterated_power=2,
			batch_size =None,random_state=None)
	famd.fit(X)
	X_t=famd.transform(X)
	corr_df=famd.column_correlation(X,same_input=True)



