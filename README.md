<img width="898" height="741" alt="Screenshot From 2026-05-09 12-07-40" src="https://github.com/user-attachments/assets/0424c012-63bb-4d44-89f6-d5e92bfc799d" /># 🌦️ ClimateCast AI

AI-powered weather time series forecasting dashboard built using Python, Prophet, Plotly, and Streamlit.

ClimateCast AI analyzes historical weather patterns and predicts future weather trends using time series forecasting techniques.

---
# 🚀 Live Demo

🔗 [Launch ClimateCast AI](https://your-streamlit-link.streamlit.app)

---
# 🚀 Features

* 📈 Weather forecasting using Facebook Prophet
* 🌡️ Forecast temperature trends
* 📅 Predict future weather for upcoming days
* 📊 Interactive visualizations with Plotly
* 📥 Download forecast results as CSV
* 🌙 Modern dark-themed dashboard
* 🧠 Confidence interval prediction bands
* ⚡ Streamlit-powered interactive UI

---

# 🧠 Tech Stack

| Layer             | Technology                   |
| ----------------- | ---------------------------- |
| Language          | Python                       |
| Forecasting Model | Prophet                      |
| Data Processing   | Pandas                       |
| Visualization     | Plotly                       |
| Web App           | Streamlit                    |
| Dataset           | Kaggle Daily Climate Dataset |

---

# 📂 Project Structure

```bash
climatecast-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── DailyDelhiClimateTrain.csv
│
├── src/
│   ├── preprocessing.py
│   ├── forecasting.py
│   └── visualization.py
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/climatecast-ai.git

cd climatecast-ai
```

---

## Create Virtual Environment

### Linux / MacOS

```bash
python -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📊 Dataset

This project uses the Daily Climate Time Series dataset from Kaggle.

Dataset contains:

* Date
* Mean Temperature
* Humidity
* Wind Speed
* Pressure

---

# 📈 Forecasting Approach

ClimateCast AI uses Facebook Prophet for time series forecasting.

The forecasting model learns:

* trend patterns
* yearly seasonality
* weekly seasonality
* uncertainty intervals

Forecasting concept:

```math
y(t)=g(t)+s(t)+\epsilon_t
```

Where:

* (g(t)) = trend
* (s(t)) = seasonality
* (\epsilon_t) = random noise

---

# 🖼️ Dashboard Preview

(Add your dashboard screenshot here)

Example:

```markdown
![Dashboard Preview](images/dashboard.png)
```

---

# 📥 Example Workflow

1. Upload weather CSV file
2. Select weather parameter
3. Choose forecast horizon
4. Generate future forecast
5. Visualize predictions
6. Download forecast CSV

---

# 🎯 Future Improvements

* 🌍 Multi-city weather forecasting
* ☁️ Real-time weather API integration
* 🚨 Anomaly detection
* 📡 Live forecasting updates
* 🧠 LSTM / Deep Learning forecasting
* 📈 Advanced climate analytics

---

# 🤝 Contributing

Contributions, ideas, and improvements are welcome.

Feel free to fork the repository and submit pull requests.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Built by Abhimanyu

Passionate about AI, Machine Learning, Forecasting Systems, and Full Stack Development.
