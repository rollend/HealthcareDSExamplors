# HealthcareDSExamplors
 # Disclaimer: No patient data included, NHS project invovling patient data only provide screenshot to show the project <br/>
## Projects: <br/>
 1.Did not attend (DNA) [No data and code included] <br/>
  -Briefing: Predict patient will attend his/her appoint or not, if not then intervention required. Current intervention message/phone call;<br/>
  -Below diagram shows the initial analysis over patient classes:<br/>
  ![Patient Clusters](/DNA/Patient%20Visits%20LDA_2015_4.png)<br/>
  <br/>
 2. Stranded Patient/Length of Stay (LOS)/Delay transfer of care (DTOC) [No data and code included]<br/>
  -Briefing: Predict whether the patient will be stranded (>7 days), forecasting his/her LOS, predict whether the patient will be delayed or not; These three projects are all derived from similar patient data, the important requirement is predict/forcast as early as possible. We have explored using SNOMED, ICD10 code as well as blood test results. Finally we are using accute data, and first three days of admission with blood test results to make early prediction; <br/>
  -Below diagram shows the exploration of ICD10 code classifies different patients; <br/>
  ![Patient Clusters ICD10](/StrandedLOSDTOC/100cluster.png)<br/>
  <br/>
 3. SHMI [All public data and code included]<br/>
  -Briefing: Forecasting the trend of SHMI from history data, challange is that there are more than 6000 thounds predictors, each has one year monthly time series data. Which model could do better, will not explode and where is the cap?; <br/> 
  <br/>
 4. Covid-19-Kaggle [Kaggle train data excluded; public data and code included]<br/>
  -Briefing: Global confirmed cases and fatalities, daily time series data (23.Jan.2020 - 18.Mar.2020). Forecasting next certain period of date (18.Mar.2020 - 13.Apr.2020); <br/>
  -Results: Performance in the middle range of competition.<br/>
 <br/>
 5.Covid-19-Local [No data or code included]<br/>
  -Briefing: Forecast local hospital covid cases. Only as indicator; <br/>
  -Status: Initialized as SIR model, next integrate with DL/GraphNN;<br/>
  -Below diagram shows the SIR: <br/>
  ![SIR](/covid-19-local/Capture.PNG)<br/>
  <br/>
 6.AdvancingAnalytics [No data or code included] <br/>
  -Briefing: Advancing Healthcare Analytics Capacity;<br/>
  -Below diagram shows the Inforgraphic/Visualization: <br/>
  ![Poster](/AdvancingAnalytics/tech-poster-final-draft-[JRG_120417].pdf)<br/>
  
 
