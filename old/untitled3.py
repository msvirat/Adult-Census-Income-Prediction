# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 22:42:03 2021

@author: Sathiya vigraman M
"""

    final = pd.DataFrame(columns=['fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hpw', 'age_Middle-age', 'age_Senior', 'age_Old', 'workclass_Private', 'workclass_Self', 'workclass_Without-pay', 'education_College', 'education_Doctorate', 'education_Masters', 'education_Prof-school', 'marital_status_married', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving', 'relationship_Others', 'race_Other', 'race_White', 'sex_Male', 'country_EMEA', 'country_LATAM', 'country_NAM', 'country_United-States' ])    
    for col in final.columns:
        if col not in df_new.columns:
            final[col] = 0
        else:
            final[col] = df_new[col]
