# 🌦️ Pakistan Climate Dashboard

A comprehensive data visualization dashboard for analyzing climate data across Pakistan's districts. This interactive web application provides insights into temperature, rainfall, drought risk, and flood risk patterns throughout the country.

## 🚀 Features

### 📊 Interactive Visualizations
- **Choropleth Maps**: Geographic distribution of climate variables across Pakistan's districts
- **Statistical Charts**: Histograms, box plots, and distribution analysis
- **Correlation Analysis**: Heatmaps showing relationships between climate variables
- **Scatter Plots**: Interactive plots exploring variable relationships

### 🎯 Key Metrics
- Average, maximum, minimum, and standard deviation for each climate variable
- Top and bottom 10 districts for each metric
- Risk analysis for drought and flood-prone areas

### 🔍 Data Exploration
- Search functionality to find specific districts
- Paginated data tables with customizable page sizes
- Interactive filtering and sorting capabilities

### 🎨 User Experience
- Beautiful, modern UI with gradient cards and custom styling
- Responsive design that works on different screen sizes
- Intuitive sidebar controls for variable selection and color schemes
- Real-time data updates and interactive charts

## 📈 Climate Variables Analyzed

1. **Average Temperature (°C)**: Mean temperature across districts
2. **Average Rainfall (mm)**: Precipitation patterns
3. **Drought Risk Index**: Vulnerability to drought conditions
4. **Flood Risk Index**: Susceptibility to flooding events

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project-Pakistan
   ```

2. **Navigate to the source directory**
   ```bash
   cd src
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

## 📁 Project Structure

```
Project Pakistan/
├── README.md
├── src/
│   ├── app.py                 # Main dashboard application
│   ├── requirements.txt       # Python dependencies
│   └── data/
│       ├── Sample_District_Climate_Data.csv    # Climate data
│       └── geoBoundaries-PAK-ADM2 (1).geojson  # Geographic boundaries
```

## 🎮 How to Use

### Dashboard Navigation

1. **Variable Selection**: Use the sidebar to choose which climate variable to visualize
2. **Color Schemes**: Select different color scales for the maps and charts
3. **Interactive Maps**: Hover over districts to see detailed information
4. **Chart Exploration**: Click and drag on charts for zoom and pan functionality

### Key Sections

- **🗺️ Geographic Distribution**: Interactive choropleth maps showing spatial patterns
- **📊 Statistical Analysis**: Distribution plots and box plots for statistical insights
- **🔗 Correlation Analysis**: Heatmaps and scatter plots showing variable relationships
- **⚠️ Risk Analysis**: Identification of high-risk districts for drought and flooding
- **🔍 Data Explorer**: Search and browse the complete dataset

## 📊 Data Sources

- **Climate Data**: Sample district-level climate data including temperature, rainfall, and risk indices
- **Geographic Data**: GeoJSON file containing Pakistan's administrative district boundaries

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **Visualization**: Plotly (Interactive charts and maps)
- **Data Processing**: Pandas, NumPy
- **Geospatial**: GeoPandas, Shapely
- **Mapping**: Folium (for additional map features)

## 🎨 Customization

### Adding New Variables
To add new climate variables:
1. Update the CSV file with new columns
2. Modify the variable selection in `app.py`
3. Add appropriate visualization functions

### Styling Changes
The dashboard uses custom CSS for styling. Modify the CSS section in `app.py` to change:
- Colors and gradients
- Font sizes and styles
- Layout spacing
- Card designs

### Map Customization
- Change map center coordinates in the `create_choropleth_map` function
- Modify zoom levels and map styles
- Add custom markers or overlays

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Missing dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Data loading errors**
   - Ensure data files are in the correct `src/data/` directory
   - Check file permissions and paths

4. **Map not displaying**
   - Verify internet connection (required for map tiles)
   - Check if GeoJSON file is properly formatted

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Time series analysis
- [ ] Export functionality for charts and data
- [ ] Additional map layers (terrain, satellite)
- [ ] Machine learning predictions
- [ ] Multi-language support
- [ ] Mobile-responsive optimizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Built with ❤️ for climate data analysis in Pakistan**

