#################
import streamlit as st
import pickle
import numpy as np
from sklearn.pipeline import Pipeline


headers={
  "autherization":st.secrets["auth_token"]
  "content-type":"application/python"
} 

# import the model
pipe = pickle.load(open('model_bet_1.pkl', 'rb'))
bet = pickle.load(open('bet_1.pkl', 'rb'))

st.title("Fraud Detection")

User_ID =  st.number_input('User ID',min_value=1,max_value=100000000000000)
inactive_days = st.number_input('Inactive days')
Country_Name = st.selectbox('Country_Name', ['Afghanistan','Albania','Algeria','Andorra','Angola','Antigua & Deps','Argentina',
 'Armenia','Australia','Austria','Azerbaijan','Bahamas','Bahrain','Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bhutan',
 'Bolivia','Bosnia Herzegovina','Botswana','Brazil','Brunei','Bulgaria','Burkina','Burundi','Cambodia','Cameroon','Canada','Cape Verde',
 'Central African Rep','Chad','Chile','China','Colombia','Comoros','Congo','Congo {Democratic Rep}','Costa Rica','Croatia','Cuba',
 'Cyprus','Czech Republic','Denmark','Djibouti','Dominica','Dominican Republic','East Timor','Ecuador','Egypt','El Salvador','Equatorial Guinea',
 'Eritrea','Estonia','Ethiopia','Fiji','Finland','France','Gabon','Gambia','Georgia','Germany','Ghana','Greece','Grenada','Guatemala','Guinea',
 'Guinea-Bissau','Guyana','Haiti','Honduras','Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland {Republic}','Israel','Italy',
 'Ivory Coast','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati','Korea North','Korea South','Kosovo','Kuwait','Kyrgyzstan',
 'Laos','Latvia','Lebanon','Lesotho','Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi',
 'Malaysia','Maldives','Mali','Malta','Marshall Islands','Mauritania','Mauritius','Mexico','Micronesia','Moldova','Monaco','Mongolia','Montenegro',
 'Morocco','Mozambique','Myanmar, {Burma}','Namibia','Nauru','Nepal','Netherlands','New Zealand','Nicaragua','Niger','Nigeria','Norway',
 'Oman','Pakistan','Palau','Panama','Papua New Guinea','Paraguay','Peru','Philippines','Poland','Portugal','Qatar','Romania','Russian Federation',
 'Rwanda','St Kitts & Nevis','St Lucia','Saint Vincent & the Grenadines','Samoa','San Marino','Sao Tome & Principe','Saudi Arabia','Senegal','Serbia','Seychelles','Sierra Leone','Singapore',
 'Slovakia','Slovenia','Solomon Islands','Somalia','South Africa','South Sudan','Spain','Sri Lanka','Sudan','Suriname','Swaziland',
 'Sweden','Switzerland','Syria','Taiwan','Tajikistan','Tanzania','Thailand','Togo','Tonga','Trinidad & Tobago','Tunisia','Turkey',
 'Turkmenistan','Tuvalu','Uganda','Ukraine','United Arab Emirates','United Kingdom','United States of America','Uruguay',
 'Uzbekistan','Vanuatu','Vatican City','Venezuela','Vietnam','Yemen','Zambia','Zimbabwe'])

banned_country=['Qatar','Argentina','United Arab Emirates','Brunei','Cambodia','North Korea','Japan','Singapore','Lebanon','Algeria','Iran',
 'Bahrain','Jordan','Kuwait','Libya','Oman','Saudi Arabia','Tunisia','Yemen','Cambodia','Syria','Cuba','Vietnam','Malaysia','Russian Federation']

Gender = st.selectbox('Gender', bet['Gender'].unique())
Year_of_Birth = Year_of_Birth = st.number_input('Year_of_Birth')
age_at_registration = st.number_input('age_at_registration',min_value=0,max_value=125)
KYC_Status = st.selectbox('KYC_Status', bet['KYC_Status'].unique())
sum_stakes_fixedodds = st.number_input('Sum Stakes Fixedodds',min_value=0,max_value=10000000)
sum_bets_fixedodds = st.number_input('sum bets fixedodds',min_value=0,max_value=1000000)
bettingdays_fixedodds = st.number_input('betting days fixedodds',min_value=0,max_value=40000)
duration_fixedodds = st.number_input('duration fixedodds',min_value=1,max_value=40000)
frequency_fixedodds_MN = st.number_input('frequency fixedodds',min_value=0.000,max_value=1.0000,step=0.0001)
bets_per_day_fixedodds_LM = st.number_input('bets per day fixedodds',min_value=0,max_value=10000)
euros_per_bet_fixedodds_KL = st.number_input('euros per bet fixedodds', min_value=0,max_value=100000)
net_loss_fixedodds = st.number_input('net loss fixedodds',  min_value=-100000,max_value=100000)
mode_of_payment = st.selectbox('mode_of_payment', bet['mode_of_payment'].unique())
bonus_type = st.selectbox('bonus_type',['Signing','Refferal'])
Ref_ID = st.number_input('Referral ID',min_value=0,max_value=100000000000000)
bonus_amount = st.number_input('bonus_amount',min_value=50,max_value=10000000)
wallet_amount = st.number_input('wallet_amount',min_value=0,max_value=10000000)
self_exclusive_status = st.selectbox('self_exclusive_status',  bet['self_exclusive_status'].unique())


if st.button('Fraud Status'):
    query = np.array(
        [User_ID,inactive_days, Country_Name, Gender, Year_of_Birth,
               age_at_registration, KYC_Status, sum_stakes_fixedodds,
               sum_bets_fixedodds, bettingdays_fixedodds, duration_fixedodds,
               frequency_fixedodds_MN, bets_per_day_fixedodds_LM,
               euros_per_bet_fixedodds_KL, net_loss_fixedodds, mode_of_payment,
               bonus_type, Ref_ID, bonus_amount, wallet_amount,
               self_exclusive_status])
    
    if Country_Name in banned_country:
        st.title("Fraud Status = Fraudulent")
    else:
        if age_at_registration <18:
            st.title("Fraud Status = Fraudulent") 
        else:    
            if self_exclusive_status == 1:
                st.title("Fraud Status = Fraudulent")
            else:
                if KYC_Status=="Rejected":
                   st.title("Fraud Status = Fraudulent")
                else:
                    query = query.reshape(1, 21)
                    if pipe.predict(query)[0] == 1:
                        st.title("Fraud Status = Fraudulent")
                    else:
                       st.title("Fraud Status = Non Fraudulent")
        
                       
                
                   
             

