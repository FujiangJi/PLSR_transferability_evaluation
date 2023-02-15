import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import verde as vd

def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a
def nse(predictions, targets):
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    
    m, p = x.shape
    _, h = t.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

def random_CV(X,y,tr,n_splits):
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y_test[tr].tolist())
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

        pred_list.extend(a)
        test_list.extend(b)

    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    xx = pred_list[pred_list>0]
    yy = test_list[pred_list>0]
    
    df = pd.DataFrame([xx,yy]).T
    df.columns = ['predicted','observed']
    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    
    df.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold random CV_df.csv',index = False)
    performance.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold random CV_performance.csv',index = False)
    
    return [df, performance]

def spatial_CV(X,y,tr,n_splits):
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    coordinates = (y.longitude, y.latitude)
    kfold = vd.BlockKFold(n_splits = n_splits, spacing=2.0, shuffle=True, random_state=5)
    feature_matrix = np.transpose(coordinates)
    balanced = kfold.split(feature_matrix)
    
    for train_index,test_index in balanced:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y_test[tr].tolist())
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

        pred_list.extend(a)
        test_list.extend(b)

    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    xx = pred_list[pred_list>0]
    yy = test_list[pred_list>0]
    
    df = pd.DataFrame([xx,yy]).T
    df.columns = ['predicted','observed']
    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    
    df.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold spatial CV_df.csv',index = False)
    performance.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold spatial CV_performance.csv',index = False)
    
    return [df, performance]

def leave_one_out_CV(X,y,tr):
    res = pd.DataFrame(np.zeros(shape = (0,6)),columns = ['training_site','testing_site','R2','RMSE','NRMSE','NSE'])
    sites = []
    for i in y['Site_num'].unique():
        df = y[y['Site_num'] == i]
        if len(df) > 100:
            sites.append(i)
    sites = np.array(sites)
    
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    vip_socre = pd.DataFrame(np.zeros(shape = (X.shape[1], len(sites))),columns = sites)
    plsr_coef = pd.DataFrame(np.zeros(shape = (X.shape[1], len(sites))),columns = sites)
    
    loo = LeaveOneOut()
    
    for test_index, train_index in loo.split(sites):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_sites = sites[train_index]
        test_sites = sites[test_index]
        
        for j in test_sites:
            df_test = y[y['Site_num']== j]
            y_test = pd.concat([y_test,df_test])
        
        y_train = y[y['Site_num']== train_sites[0]]
        X_train = X.iloc[y_train.index]
        X_test = X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        
        vvv = vip(X_train, y_train[tr], pls)
        coef = abs(pls.coef_.reshape(-1,))
        coef = (coef-coef.min())/(coef.max()-coef.min())
        vip_socre[train_sites[0]] = vvv
        plsr_coef[train_sites[0]] = coef
        
        pred = pls.predict(X_test)
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)
        
        new_df[['Site_num','pred',tr]].to_csv('1_results/'+tr+'_train_site_'+train_sites[0]+'_df.csv',index = False)
        
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for k in new_df['Site_num'].unique():
            data = new_df[new_df['Site_num'] == k]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['Site_num'].unique()[0],k, R_2,r_mse,n_rmse,N_S_E]).reshape(1,6),columns = ['training_site','testing_site','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = sites
    
    res.to_csv('1_results/'+tr+'_Leave_one_site_out_accuracy.csv',index = False)
    performance.to_csv('1_results/'+tr+'_Leave_one_site_out_overall_accuracy.csv')
    vip_socre.to_csv('1_results/'+tr+'_plsr_vip_socre.csv',index = False)
    plsr_coef.to_csv('1_results/'+tr+'_plsr_coefficients.csv',index = False)
    
    return [res,performance]

def site_extropolation(X,y,tr):
    
    s = pd.DataFrame(y['Site_num'].value_counts())#.index.tolist()
    s = s[s['Site_num']>30]
    s = s.index.tolist()
    s.reverse()
    s = np.array(s)
    
    R2_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    RMSE_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nrmse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    
    for j in s:
        target_site = j
        source_site = s[s!=target_site]
    
        accu = []
        RMSE = []
        NRMSE = []
        NSE = []
    
        for k in range(1,len(source_site)+1):
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            for i in range(k):
                df_train = y[y['Site_num']== source_site[i]]
                y_train = pd.concat([y_train,df_train])

            y_test = y[y['Site_num']== target_site]

            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]

            pls = PLSRegression(n_components=30)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            a = np.array(pred.reshape(-1,).tolist())
            b = np.array(y_test[tr].tolist())

            R2 = rsquared(a,b)
            rmse = np.sqrt(mean_squared_error(a,b))
            nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
            N_SE = nse(a, b)

            accu.append(R2)
            RMSE.append(rmse)
            NSE.append(N_SE)
            NRMSE.append(nrmse)
        
        accu = pd.DataFrame(accu,columns = [j])
        RMSE = pd.DataFrame(RMSE,columns = [j])
        NRMSE = pd.DataFrame(NRMSE,columns = [j])
        NSE = pd.DataFrame(NSE,columns = [j])
        
        R2_ex_frame = pd.concat([R2_ex_frame,accu],axis = 1)
        RMSE_ex_frame = pd.concat([RMSE_ex_frame,RMSE],axis = 1)
        nrmse_ex_frame = pd.concat([nrmse_ex_frame,NRMSE],axis = 1)
        nse_ex_frame = pd.concat([nse_ex_frame,NSE],axis = 1)
    
    R2_ex_frame.to_csv('1_results/'+tr+'_'+'R2 extrapolation.csv',index = False)
    RMSE_ex_frame.to_csv('1_results/'+tr+'_'+'RMSE extrapolation.csv',index = False)
    nrmse_ex_frame.to_csv('1_results/'+tr+'_'+'nrmse extrapolation.csv',index = False)
    nse_ex_frame.to_csv('1_results/'+tr+'_'+'nse extrapolation.csv',index = False)
    return[R2_ex_frame,RMSE_ex_frame,nrmse_ex_frame,nse_ex_frame]

def leave_one_PFT_out(X,y,tr):
    res = pd.DataFrame(np.zeros(shape = (0,6)),columns = ['training_PFT','testing_PFTs','R2','RMSE','NRMSE','NSE'])
    PFTs = y['PFT'].unique().tolist()
    PFTs = np.array([i for i in PFTs if pd.isnull(i) == False and i != 'nan'])
    
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    loo = LeaveOneOut()
    for test_index, train_index in loo.split(PFTs):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_sets = PFTs[train_index]
        test_sets = PFTs[test_index]
        
        for j in test_sets:
            df_test = y[y['PFT']== j]
            y_test = pd.concat([y_test,df_test])
            
        y_train = y[y['PFT']== train_sets[0]]
        X_train = X.iloc[y_train.index]
        X_test = X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)
        
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)
        
        new_df[['PFT','pred',tr]].to_csv('1_results/'+tr+'_train_PFT_'+train_sets[0]+'_df.csv',index = False)
        
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for k in new_df['PFT'].unique():
            data = new_df[new_df['PFT'] == k]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['PFT'].unique()[0],k, R_2,r_mse,n_rmse,N_S_E]).reshape(1,6),columns = ['training_PFT','testing_PFTs','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = PFTs
    
    res.to_csv('1_results/'+tr+'_Leave_one_PFT_out_accuracy.csv',index = False)
    performance.to_csv('1_results/'+tr+'_Leave_one_PFT_out_overall_accuracy.csv')
    return [res,performance]

def random_temporal_CV(X,y,tr,n_splits,dataset_num):
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y_test[tr].tolist())
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

        pred_list.extend(a)
        test_list.extend(b)

    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    xx = pred_list[pred_list>0]
    yy = test_list[pred_list>0]
    
    df = pd.DataFrame([xx,yy]).T
    df.columns = ['predicted','observed']
    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    
    df.to_csv('1_results/'+dataset_num+'_'+tr+'_'+str(n_splits)+'fold temporal_random CV_df.csv',index = False)
    performance.to_csv('1_results/'+dataset_num+'_'+tr+'_'+str(n_splits)+'fold temporal_random CV_performance.csv',index = False)
    
    return [df, performance]

def temporal_CV(X,y,tr,n_splits,dataset_num):
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    time = y['Sample date'].unique()
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(time):
        train_time = time[train_index]
        test_time = time[test_index]
        
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
    
        for i in train_time:
            temp = y[y['Sample date'] == i]
            y_train = pd.concat([y_train,temp])
        for i in test_time:
            temp = y[y['Sample date'] == i]
            y_test = pd.concat([y_test,temp])
            
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y_test[tr].tolist())
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

        pred_list.extend(a)
        test_list.extend(b)

    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    xx = pred_list[pred_list>0]
    yy = test_list[pred_list>0]
    
    df = pd.DataFrame([xx,yy]).T
    df.columns = ['predicted','observed']
    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    
    df.to_csv('1_results/'+dataset_num+'_'+tr+'_'+str(n_splits)+'fold temporal CV_df.csv',index = False)
    performance.to_csv('1_results/'+dataset_num+'_'+tr+'_'+str(n_splits)+'fold temporal CV_performance.csv',index = False)
    
    return [df, performance]

def leave_one_season_out_CV(X,y,tr,dataset_num):
    res = pd.DataFrame(np.zeros(shape = (0,6)),columns = ['training_season','testing_season','R2','RMSE','NRMSE','NSE'])
    time = y['season'].unique()
    
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    
    loo = LeaveOneOut()
    for test, train in loo.split(time):
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        train_time = time[train]
        test_time = time[test]

        for j in test_time:
            df_test = y[y['season'] == j]
            y_test = pd.concat([y_test,df_test])
        
        y_train = y[y['season']== train_time[0]]
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)
        
        pred = pd.DataFrame(pred,columns = ['pred'])
        pred.reset_index(drop = True, inplace = True)
        y_test.reset_index(drop = True, inplace = True)
        new_df = pd.concat([pred,y_test],axis = 1)
        
        new_df[['season','Sample date','pred',tr]].to_csv('1_results/'+tr+'_'+dataset_num+'_train_season_'+train_time[0]+'_df.csv',index = False)
        
        a = new_df['pred']
        b = new_df[tr]
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)
        
        for k in new_df['season'].unique():
            data = new_df[new_df['season'] == k]

            R_2 = rsquared(data['pred'],data[tr])
            r_mse = np.sqrt(mean_squared_error(data['pred'],data[tr]))
            n_rmse = np.sqrt(mean_squared_error(data['pred'],data[tr]))/(data[tr].max()-data[tr].min())
            N_S_E = nse(data['pred'],data[tr])
            
            temp = pd.DataFrame(np.array([y_train['season'].unique()[0],k, R_2,r_mse,n_rmse,N_S_E]).reshape(1,6),columns = ['training_season','testing_season','R2','RMSE','NRMSE','NSE'])
            res = pd.concat([res,temp])

    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    performance.index = time
    
    res.to_csv('1_results/'+tr+'_'+dataset_num+'_Leave_one_season_out_accuracy.csv',index = False)
    performance.to_csv('1_results/'+tr+'_'+dataset_num+'_Leave_one_season_out_overall_accuracy.csv')
    return [res,performance]

def PFT_extropolation(X,y,tr):
    
    s = pd.DataFrame(y['PFT'].value_counts())
    s = s[s['PFT']>2]
    s = s.index.tolist()
    s.reverse()
    s = np.array(s)
    
    R2_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    RMSE_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nrmse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    nse_ex_frame = pd.DataFrame(np.zeros(shape = (len(s)-1,0)))
    
    for j in s:
        target_site = j
        source_site = s[s!=target_site]
    
        accu = []
        RMSE = []
        NRMSE = []
        NSE = []
    
        for k in range(1,len(source_site)+1):
            y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
            for i in range(k):
                df_train = y[y['PFT']== source_site[i]]
                y_train = pd.concat([y_train,df_train])

            y_test = y[y['PFT']== target_site]

            X_train = X.iloc[y_train.index]
            X_test = X.iloc[y_test.index]

            pls = PLSRegression(n_components=30)
            pls.fit(X_train, y_train[tr])
            pred = pls.predict(X_test)

            a = np.array(pred.reshape(-1,).tolist())
            b = np.array(y_test[tr].tolist())

            R2 = rsquared(a,b)
            rmse = np.sqrt(mean_squared_error(a,b))
            nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
            N_SE = nse(a, b)

            accu.append(R2)
            RMSE.append(rmse)
            NSE.append(N_SE)
            NRMSE.append(nrmse)
        
        accu = pd.DataFrame(accu,columns = [j])
        RMSE = pd.DataFrame(RMSE,columns = [j])
        NRMSE = pd.DataFrame(NRMSE,columns = [j])
        NSE = pd.DataFrame(NSE,columns = [j])
        
        R2_ex_frame = pd.concat([R2_ex_frame,accu],axis = 1)
        RMSE_ex_frame = pd.concat([RMSE_ex_frame,RMSE],axis = 1)
        nrmse_ex_frame = pd.concat([nrmse_ex_frame,NRMSE],axis = 1)
        nse_ex_frame = pd.concat([nse_ex_frame,NSE],axis = 1)
    
    R2_ex_frame.to_csv('1_results/'+tr+'_'+'R2 PFT_extrapolation.csv',index = False)
    RMSE_ex_frame.to_csv('1_results/'+tr+'_'+'RMSE PFT_extrapolation.csv',index = False)
    nrmse_ex_frame.to_csv('1_results/'+tr+'_'+'nrmse PFT_extrapolation.csv',index = False)
    nse_ex_frame.to_csv('1_results/'+tr+'_'+'nse PFT_extrapolation.csv',index = False)
    return[R2_ex_frame,RMSE_ex_frame,nrmse_ex_frame,nse_ex_frame]

def cross_PFT(X,y,tr,n_splits):
    accu = []
    RMSE = []
    NRMSE = []
    NSE = []
    pred_list = []
    test_list = []
    
    time = y['PFT'].unique()
    time = np.array([i for i in time if pd.isnull(i) == False and i != 'nan'])
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(time):
        train_time = time[train_index]
        test_time = time[test_index]
        
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
    
        for i in train_time:
            temp = y[y['PFT'] == i]
            y_train = pd.concat([y_train,temp])
        for i in test_time:
            temp = y[y['PFT'] == i]
            y_test = pd.concat([y_test,temp])
            
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        pls = PLSRegression(n_components=30)
        pls.fit(X_train, y_train[tr])
        pred = pls.predict(X_test)

        a = np.array(pred.reshape(-1,).tolist())
        b = np.array(y_test[tr].tolist())
        
        R2 = rsquared(a,b)
        rmse = np.sqrt(mean_squared_error(a,b))
        nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
        N_SE = nse(a, b)
        
        accu.append(R2)
        RMSE.append(rmse)
        NSE.append(N_SE)
        NRMSE.append(nrmse)

        pred_list.extend(a)
        test_list.extend(b)

    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    xx = pred_list[pred_list>0]
    yy = test_list[pred_list>0]
    
    df = pd.DataFrame([xx,yy]).T
    df.columns = ['predicted','observed']
    performance = pd.DataFrame([accu,RMSE, NRMSE, NSE]).T
    performance.columns = ['R2','RMSE','NRMSE','NSE']
    
    df.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold PFTs CV_df.csv',index = False)
    performance.to_csv('1_results/'+tr+'_'+str(n_splits)+'fold PFTs CV_performance.csv',index = False)
    
    return [df, performance]


def cross_sites_PFTs(X,y,tr):
    PFTs = y['PFT'].unique().tolist()
    PFTs = np.array([i for i in PFTs if pd.isnull(i) == False and i != 'nan'])

    col = ['PFTs','R2_mean',' R2_std','RMSE_mean','RMSE_std','NSE_mean','NSE_std','NRMSE_mean','NRMSE_std']
    total_accuracy = pd.DataFrame(np.zeros(shape = (0,len(col))),columns = col)

    for pfts in PFTs:
        y_pft = y[y['PFT'] == pfts]
        X_pft = X.loc[y_pft.index]
        y_pft.reset_index(drop = True, inplace = True)
        X_pft.reset_index(drop = True, inplace = True)

        sites = []
        for i in y_pft['Site_num'].unique():
            df = y_pft[y_pft['Site_num'] == i]
            if len(df) > 30:
                sites.append(i)
        sites = np.array(sites)

        if len(sites)>1:
            accu = []
            RMSE = []
            NRMSE = []
            NSE = []

            loo = LeaveOneOut()
            for test_index, train_index in loo.split(sites):
                train_sites = sites[train_index]
                test_sites = sites[test_index]

                y_train = y_pft[y_pft['Site_num']== train_sites[0]]
                X_train = X_pft.iloc[y_train.index]

                pls = PLSRegression(n_components=30)
                pls.fit(X_train, y_train[tr])

                accu1 = []
                RMSE1 = []
                NRMSE1 = []
                NSE1 = []

                for i in test_sites:
                    y_test = y_pft[y_pft['Site_num']== i]
                    X_test = X_pft.iloc[y_test.index]

                    pred = pls.predict(X_test)
                    pred = pd.DataFrame(pred,columns = ['pred'])
                    pred.reset_index(drop = True, inplace = True)
                    y_test.reset_index(drop = True, inplace = True)
                    new_df = pd.concat([pred,y_test],axis = 1)

                    a = new_df['pred']
                    b = new_df[tr]

                    R2 = rsquared(a,b)
                    rmse = np.sqrt(mean_squared_error(a,b))
                    nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
                    N_SE = nse(a, b)

                    accu1.append(R2)
                    RMSE1.append(rmse)
                    NSE1.append(N_SE)
                    NRMSE1.append(nrmse)

                accu.extend(accu1)
                RMSE.extend(RMSE1)
                NSE.extend(NSE1)
                NRMSE.extend(NRMSE1)

            R2_mean = np.array(accu).mean()
            R2_std = np.array(accu).std()
            RMSE_mean = np.array(RMSE).mean()
            RMSE_std = np.array(RMSE).std()
            NSE_mean = np.array(NSE).mean()
            NSE_std = np.array(NSE).std()
            NRMSE_mean = np.array(NRMSE).mean()
            NRMSE_std = np.array(NRMSE).std()

            accuracy = pd.DataFrame(np.array([pfts,R2_mean, R2_std,RMSE_mean,RMSE_std,NSE_mean,NSE_std,NRMSE_mean,NRMSE_std])).T
            accuracy.columns = col
            total_accuracy = pd.concat([total_accuracy,accuracy])
    total_accuracy.reset_index(drop = True, inplace = True)
    total_accuracy.to_csv('1_results/'+tr+'_'+'Cross sites for each PFT.csv',index = False)


def cross_sites_PFTs_new(X,y,tr):
    PFTs = y['PFT'].unique().tolist()
    PFTs = np.array([i for i in PFTs if pd.isnull(i) == False and i != 'nan'])
    col = ['PFTs','R2','RMSE','NSE','NRMSE']
    total_accuracy = pd.DataFrame(np.zeros(shape = (0,len(col))),columns = col)

    for pfts in PFTs:
        y_pft = y[y['PFT'] == pfts]
        X_pft = X.loc[y_pft.index]
        y_pft.reset_index(drop = True, inplace = True)
        X_pft.reset_index(drop = True, inplace = True)

        sites = []
        for i in y_pft['Site_num'].unique():
            df = y_pft[y_pft['Site_num'] == i]
            if len(df) > 30:
                sites.append(i)
        sites = np.array(sites)

        if len(sites)>1:
            accu = []
            RMSE = []
            NRMSE = []
            NSE = []

            loo = LeaveOneOut()
            for test_index, train_index in loo.split(sites):
                train_sites = sites[train_index]
                test_sites = sites[test_index]

                y_train = y_pft[y_pft['Site_num']== train_sites[0]]
                X_train = X_pft.iloc[y_train.index]

                pls = PLSRegression(n_components=30)
                pls.fit(X_train, y_train[tr])

                accu1 = []
                RMSE1 = []
                NRMSE1 = []
                NSE1 = []

                for i in test_sites:
                    y_test = y_pft[y_pft['Site_num']== i]
                    X_test = X_pft.iloc[y_test.index]

                    pred = pls.predict(X_test)
                    pred = pd.DataFrame(pred,columns = ['pred'])
                    pred.reset_index(drop = True, inplace = True)
                    y_test.reset_index(drop = True, inplace = True)
                    new_df = pd.concat([pred,y_test],axis = 1)

                    a = new_df['pred']
                    b = new_df[tr]

                    R2 = rsquared(a,b)
                    rmse = np.sqrt(mean_squared_error(a,b))
                    nrmse = np.sqrt(mean_squared_error(a,b))/(b.max()-b.min())
                    N_SE = nse(a, b)

                    accu1.append(R2)
                    RMSE1.append(rmse)
                    NSE1.append(N_SE)
                    NRMSE1.append(nrmse)

                accu.extend(accu1)
                RMSE.extend(RMSE1)
                NSE.extend(NSE1)
                NRMSE.extend(NRMSE1)

            a = pd.DataFrame(np.array(accu),columns = ['R2'])
            b = pd.DataFrame(np.array(RMSE),columns = ['RMSE'])
            c = pd.DataFrame(np.array(NSE),columns = ['NSE'])
            d = pd.DataFrame(np.array(NRMSE),columns = ['NRMSE'])
            temp = pd.concat([a,b,c,d],axis = 1)
            temp['PFTs'] = pfts
            total_accuracy = pd.concat([total_accuracy,temp])
    total_accuracy.reset_index(drop = True, inplace = True)
    total_accuracy.to_csv('1_results/'+tr+'_'+'Cross sites for each PFT_new.csv',index = False)
