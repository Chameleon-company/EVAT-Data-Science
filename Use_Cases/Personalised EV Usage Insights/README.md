# Personalised EV Usage Insights – Data & Dashboard Prototype
The Personalised EV Usage Insights project provides drivers with tailored insights based on their driving patterns and behaviours. It allows drivers to see how their fuel consumption and spending compare with similar drivers, and highlights the potential cost savings of switching to an electric vehicle (EV).

By making these comparisons transparent, the project helps drivers understand their current fuel expenditure, identify opportunities to reduce costs, and consider the environmental benefits of lowering fuel emissions through EV adoption.

The project combines data collection, data-driven clustering, dynamic integration along with PowerBI dashboard development.

Visit: https://deakin365.sharepoint.com/sites/Chameleon2/SitePages/Personalized-EV-Usage-Insights.aspx for a detailed workflow.

## Workflow
This section details the end-to-end workflow and technical methodology used to collect, process, analyse, and visualise personalised EV usage data, highlighting the tools, algorithms, and integrations that enable real-time insights for users.

1. User inputs data to the following Google Form: https://docs.google.com/forms/d/e/1FAIpQLScLn9syI05fYg5-6o7CZNj0DS_M0HYoQ1uCe_ZPUws4d8hXDg/viewform.
2. Responses can be viewed at: https://docs.google.com/spreadsheets/d/1gvyLh86Qsm9FNa_ihCf496818rh5dN-9GBnMVO5ud_A/edit?usp=sharing.
3. The **'dataset'** folder contains the synthetic data file created via **Mockaroo** to simulate real Australian driver behaviour.
4. Copy **'evat_webhook'** files to your Git and deploy to Render as a web service (Node.js/Express Webhook API). Ensure you change the **'mongoWebhookURL'** to the URL provided by Render.
5. Use **UpTimeRobot** to keep your Web Service live (e.g. ping web service every 10 minutes).
6. The **'Google Form Apps Script.txt'** file in the repository can be copied to Google Apps Script to push data to the EVAT MongoDB database via the Web Service in Render.
7. K-Prototype algorithm was used in Google Colab (Python) to train the data. Python code available here: https://colab.research.google.com/drive/17DO_OOCeumWVqrowVK3hI_G5Nr5PcyCK?usp=sharing
8. This generated 4 distinct User Segments which was downloaded and saved in the **'dataset'** folder.
9. To ensure future user responses are clustered automatically based on the trained data, copy **'evat_cluster_service'** files to Git and deploy to Render as a web service (Python Flask microservice). The Node.js/Express backend communicates with the Flask service via Axios calls, allowing new user submissions to be assigned to clusters in real time.
10. Each submission is stored in MongoDB with an automatically attached cluster label.
11. A dashboard was developed from the cluster results. File: `Personalised EV Usage Insights.pbix`
12. Dashboard was published and embedded to Sharepoint.

## Updates
1. Prediction model now incorporates user authentication via secure GET endpoint for user-specific data
2. Posts data from MongoDB if user authentication is successful
