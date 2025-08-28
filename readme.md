🏠 House Price Predictor
Estimate residential property prices using a trained regression model with Box-Cox transformation.
Built with Python, Streamlit, and scikit-learn, this app supports both manual input and batch prediction via CSV upload.

🚀 Features
- 🔧 Manual Prediction: Enter property features and get an instant price estimate.
- 📁 Batch Prediction: Upload a CSV file with multiple property records and receive predictions for each.
- 📊 In-App Display: View predicted prices directly in the app.
- 📥 Downloadable Results: Export predictions as a CSV file.
- ✅ Robust Validation: Handles missing or malformed data gracefully.

📦 Tech Stack
- Frontend: Streamlit
- Backend: Python, scikit-learn
- Model: Regression with Box-Cox transformation
- Deployment: Streamlit Cloud

📄 Input Parameters
|  |  | 
| crime_rate |  | 
| resid_area |  | 
| air_qual |  | 
| room_num |  | 
| age |  | 
| teachers |  | 
| poor_prop |  | 
| n_hos_beds |  | 
| parks |  | 
| dist |  | 
| airport |  | 
| waterbody_River |  | 


💡 Note: All prices are predicted in USD millions.


📁 CSV Format for Batch Prediction
Your CSV should include the 13 columns listed above. Example:
crime_rate,resid_area,air_qual,room_num,age,teachers,poor_prop,n_hos_beds,parks,dist,airport,waterbody_River
0.00632,32.31,0.538,6.575,65.2,24.7,4.98,5.48,0.049,4.1,1,1
...



🛠️ How to Run Locally
git clone https://github.com/VikramNarendran/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
streamlit run app_predict.py



🌐 Live Demo
🔗 Streamlit Cloud App
(Replace with your actual deployment URL)


