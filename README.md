# Personalised EV Usage Insights – Data & Dashboard Prototype

The Personalised EV Usage Insights project provides drivers with tailored insights based on their driving patterns, fuel consumption behaviour, and EV suitability indicators. The system allows users to compare their driving characteristics with similar drivers, estimate possible EV-related savings, analyse environmental impact, and receive recommendation-based insights for EV adoption.

The project combines:
- MongoDB data integration
- Node.js backend services
- recommendation analytics
- behavioural clustering
- Power BI dashboard development
- interactive visualisation
- machine learning and data processing workflows


---

## Workflow

This section explains the complete workflow used to collect, process, analyse, and visualise personalised EV usage insights.

### 1. User Data Collection

Users provide driving-related information through the EVAT data collection workflow. The collected information includes:
- weekly driving distance
- monthly fuel spending
- fuel efficiency
- driving type
- charging preferences
- road trip behaviour
- home charging availability
- solar panel ownership
- vehicle usage characteristics
- EV-related priorities and preferences

Initial user response collection was integrated through Google Forms and backend webhook services.

---

### 2. Backend and MongoDB Integration

A Node.js and Express backend service is used to receive user submissions and process incoming responses. The backend validates user inputs and stores processed responses inside MongoDB.

MongoDB acts as the primary storage layer for:
- user submissions
- behavioural attributes
- cluster labels
- recommendation outputs
- dashboard-linked analytics data

The backend also communicates with recommendation and clustering workflows to generate additional analytical outputs for each user response.

---

### 3. Synthetic Dataset and Training Data

The project uses both synthetic and processed datasets to simulate realistic Australian driving behaviour and EV adoption scenarios.

The datasets support:
- clustering analysis
- comparative analytics
- recommendation workflows
- dashboard visualisation
- EV suitability estimation

Synthetic driver datasets were generated to improve behavioural coverage and support machine learning experimentation.

---

### 4. Driver Clustering

Behavioural clustering techniques were used to segment drivers into groups based on:
- travel behaviour
- fuel consumption
- driving distance
- road usage patterns
- charging accessibility
- vehicle usage characteristics

Example driver clusters include:
- Eco-Conscious City Drivers
- Urban Short-Trippers
- Highway Commuters
- Frequent Long-Distance Drivers

The clustering workflow enables:
- personalised comparisons
- driver segmentation
- recommendation targeting
- dashboard filtering and interaction

Each processed user submission is associated with a behavioural cluster label.

---

### 5. Recommendation Analytics Workflow

A recommendation analytics workflow was implemented to estimate EV suitability and generate sustainability-related insights.

The recommendation system analyses:
- weekly driving distance
- fuel efficiency
- charging accessibility
- driving type
- travel behaviour
- road trip frequency
- user energy usage characteristics

Based on these behavioural features, users are categorised into recommendation groups such as:
- EV Optional
- Hybrid Recommended
- Full EV Recommended

The workflow additionally estimates:
- annual fuel savings
- estimated EV charging costs
- estimated CO₂ reduction
- fuel usage reduction

Processed recommendation outputs are exported and connected to Power BI for analytical visualisation.

---

### 6. Machine Learning and Analytical Processing

Python and Jupyter Notebook workflows were used for:
- data preprocessing
- feature analysis
- behavioural clustering
- recommendation logic
- comparative analysis
- output dataset generation

Recommendation outputs are integrated into Power BI dashboards for interactive visual analytics.

The recommendation workflow supports:
- sustainability estimation
- financial estimation
- behavioural analysis
- EV suitability analytics

---

### 7. Power BI Dashboard Development

Interactive Power BI dashboards were developed to visualise personalised EV insights and recommendation analytics.

The dashboards are connected to:
- MongoDB user response datasets
- recommendation output datasets
- processed analytical outputs

The dashboard currently contains two main pages:

### Page 1 – Personalised EV Usage Insights

This dashboard provides:
- user driving statistics
- fuel spending comparison
- fuel efficiency comparison
- EV savings estimation
- driver cluster analytics
- comparative behavioural insights
- interactive filtering

Interactive slicers support filtering by:
- driver cluster
- driving type

---

### Page 2 – EV Recommendation Analytics

This dashboard provides:
- EV recommendation distribution
- estimated annual savings
- estimated EV charging costs
- estimated CO₂ reduction
- recommendation insight summaries
- recommendation-based filtering and analytics

The dashboard enables users to analyse how behavioural characteristics influence EV suitability recommendations.

---

### 8. Current Data Flow

Current system workflow:

1. User submits driving-related information
2. Backend validates and stores responses in MongoDB
3. Behavioural and recommendation processing is performed
4. Recommendation outputs and cluster labels are generated
5. Processed datasets are exported and linked to Power BI
6. Power BI dashboards visualise EV insights and recommendation analytics

---

### 9. Future Improvements

Potential future improvements include:
- real-time MongoDB to Power BI integration
- automated dashboard refresh pipelines
- advanced predictive recommendation models
- improved recommendation explainability
- enhanced dashboard interactivity
- authentication-based personalised dashboards
- deployment integration with EVAT web application
- automated retraining workflows
- cloud deployment and live analytics integration

---



## How to Run the Project

### Backend Setup

1. Navigate to the backend folder
2. Install dependencies:


npm install


3. Start backend service:


node index.js


4. Configure MongoDB connection using `.env`

---

### Machine Learning Workflow

1. Open the Jupyter Notebook workflow
2. Run notebook cells sequentially
3. Generate recommendation outputs
4. Export processed datasets if required

---

### Power BI Dashboard

1. Open `.pbix` dashboard file using Power BI Desktop
2. Refresh datasets if required
3. Ensure linked datasets and MongoDB outputs are accessible locally


---

## Notes

This project was developed to support data-driven EV adoption analysis, behavioural analytics, sustainability insights, and interactive recommendation visualisation for the EVAT platform.