# Automotive Datasets and Resources for CarSense ML Service

This comprehensive document compiles various datasets and resources for automotive diagnostics and machine learning applications, with a special focus on European vehicles common in the Romanian market.

## 1. Public OBD2 Datasets

### 1.1 Automotive OBD-II Dataset (KIT - Karlsruhe Institute of Technology)
- **Source**: [RADAR KIT Dataset](https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq)
- **Description**: CSV data containing ten vehicle signals logged via the OBD-II interface
- **Features**: Engine coolant temperature, intake manifold pressure, RPM, vehicle speed, intake air temperature, air flow rate, throttle position, ambient air temperature, accelerator pedal positions
- **Vehicles**: VW Passat
- **Collection Method**: OBD-II dongle KIWI 3 from PLX Devices with smartphone application OBD Auto Doctor (iOS)
- **Size**: 11.6 MB
- **License**: CC BY 4.0
- **Relevance**: Performance analysis, engine diagnostics, behavior modeling

### 1.2 LEVIN Vehicle Telematics Data
- **Source**: [GitHub - YunSolutions/levin-openData](https://github.com/YunSolutions/levin-openData)
- **Description**: Data collected from proprietary OBD device (LEVIN) over 4 months on 30 cars
- **Labels**: Contains DTC codes and corresponding vehicle parameters, allowing for correlation analysis between codes and vehicle conditions
- **Features**: Device ID, timestamp, trip ID, accelerometer data, GPS speed, battery voltage, coolant temperature, DTCs, engine load
- **Collection Method**: OBD data at 1Hz frequency, accelerometer data at 25Hz
- **Applications**: Driving patterns, gear detection, event detection (hard brakes, acceleration, turns, lane changes), impact detection
- **License**: MIT License

### 1.3 Driving Dataset (VW Passat)
- **Source**: [GitHub - aromanrsc/driving-ds](https://github.com/aromanrsc/driving-ds)
- **Description**: Dataset collected via OBD-II port using the OBD Fusion mobile app
- **Features**: Contains 134 features recorded every second during driving, including regular and sensitive behavior patterns
- **Paper**: "Evaluating the Privacy and Utility of Time-Series Data Perturbation Algorithms" by Adrian-Silviu Roman
- **Relevance**: Driver behavior analysis, privacy protection in vehicle data

### 1.4 Hayatu's Automotive Diagnostics Dataset
- **Source**: [GitHub - hayatu4islam/Automotive_Diagnostics](https://github.com/hayatu4islam/Automotive_Diagnostics)
- **Description**: Dataset for analysis and diagnostics of automotive performance
- **Features**: Engine coolant temperature, intake manifold pressure, RPM, vehicle speed, intake air temperature
- **Applications**: Engine diagnostics, performance optimization, anomaly detection, fuel efficiency analysis, fault detection

### 1.5 VehicalDiagnosticAlgo
- **Source**: [GitHub - prithvisekhar/VehicalDiagnosticAlgo](https://github.com/prithvisekhar/VehicalDiagnosticAlgo)
- **Description**: Collection of automotive diagnostic data with focus on low-cost hardware and open-source algorithms
- **Features**: Mobile phone sensor data, OBD2 vehicle data, and camera information
- **Applications**: Traffic analysis, vehicle health monitoring, driving behavior analysis

### 1.6 ArjanKw's OBD2 Open Data
- **Source**: [GitHub - ArjanKw/obd2-data](https://github.com/ArjanKw/obd2-data)
- **Description**: OBD2 datasets collected with the app "CarScanner"
- **Vehicles**: Includes data from Volvo V40
- **Features**: Data separated by driver/car combination
- **Applications**: Driving behavior analysis, fuel consumption analysis, tire inflation monitoring, rev matching analysis
- **License**: MIT License

### 1.7 CarOBD Dataset (Toyota Etios 2014)
- **Source**: [GitHub - eron93br/carOBD](https://github.com/eron93br/carOBD)
- **Description**: OBD2 dataset with 27 different vehicle parameters including MIL (Malfunction Indicator Lamp) status
- **Labels**: Contains parameters like "Time run with MIL on", "Time since trouble codes cleared", "Distance traveled with MIL on"
- **Collection Method**: Carloop embedded system with cellular connectivity
- **Data Categories**: Idle mode, motion mode (high speed), specific trajectories, university campus (low speed), long trips

### 1.8 OBD-II Eco-driving Assistant Dataset
- **Source**: [GitHub - Yudong-Fan/OBD-II-Eco-driving-assistant](https://github.com/Yudong-Fan/OBD-II-Eco-driving-assistant)
- **Description**: Dataset with labeled driving efficiency scores
- **Labels**: Contains efficiency classifications for different driving patterns
- **Features**: Multiple feature sets available (7, 16, and 43 features) for training different model complexities
- **Applications**: Predicting optimal driving conditions and maintenance needs

## 2. European Vehicle-Specific Resources

### 2.1 BMW Deep OBD Configurations
- **Source**: [GitHub - hanynowsky/bmwdeepobdconf](https://github.com/hanynowsky/bmwdeepobdconf)
- **Description**: BMW Deep OBD configurations for specific BMW engines including N52K and N42
- **Vehicles**: BMW E46 316ti, BMW E60 523i LCI
- **Relevance**: Deep diagnostics of BMW vehicles, sensor mapping

### 2.2 BMW-Specific OBD2 Resources
- BMW vehicles use specific diagnostic protocols and codes
- BMW-specific PIDs (Parameter IDs) often have their own proprietary codes beyond the standard OBD2 set
- BMW vehicles typically use the CAN (Controller Area Network) protocol for OBD2 communications
- The BMW diagnostic system often requires specialized tools like INPA, NCS Expert, or commercial scan tools

### 2.3 Mercedes-Benz ODX Tools
- **Source**: [GitHub - mercedes-benz/odxtools](https://github.com/mercedes-benz/odxtools)
- **Description**: Official Mercedes-Benz utilities for working with diagnostic descriptions using the ODX standard
- **Features**:
  - Parsing ODX/PDX diagnostic database files
  - Encoding/decoding diagnostic requests and responses
  - Command line tools for exploring diagnostic capabilities
  - Interactive browsing of vehicle diagnostic services
  - Decoding diagnostic sessions in real-time
- **Use Cases**:
  - Prototype development for vehicle diagnostics
  - End-of-production calibration
  - After-sales service implementation
  - Analysis of diagnostic sessions
  - Implementation of bridges to higher-level protocols

### 2.4 Dacia Diagnostic Tools
- **Source**: [iCarsoft Dacia Diagnostic Tools](https://www.icarsoft.eu/Dacia-Diagnostic-Tools)
- **Description**: Professional diagnostic tools for Dacia vehicles, including software for comprehensive diagnostics
- **Features**: Support for Dacia, Renault, Alfa Romeo, Citroën, Fiat, Peugeot
- **Available Functions**: 
  - Diagnostics
  - DPF (Diesel Particle Filter) management
  - Oil service reset
  - Bleeding functions
  - Electronic throttle control
  - Steering angle sensor calibration
  - Battery management system
  - Injector coding

### 2.5 Renault/Dacia OBD2 Diagnostic Resources
- **Source**: [YouTube - Best Renault Dacia OBD2 Diagnostic Scan Tools](https://www.youtube.com/watch?v=iwAzjd053n8)
- **Description**: Overview of available diagnostic tools specifically for Renault and Dacia vehicles
- **Features**: Comparison of different OBD2 scan tools and their compatibility with Renault/Dacia vehicles

## 3. Labeled Datasets for Predictive Maintenance

### 3.1 NHTSA's Office of Defects Investigation (ODI) Complaints Dataset
- **Source**: [Department of Transportation - NHTSA](https://catalog.data.gov/dataset/nhtsas-office-of-defects-investigation-odi-complaints)
- **Description**: Comprehensive database of vehicle owner complaints with specific defect information and outcomes
- **Labels**: Contains vehicle problems, component classifications, and resolution information
- **Features**: Vehicle make/model/year, problem components, failure descriptions, and more
- **Applications**: Mapping common failure modes to specific vehicle models and conditions
- **Update Frequency**: Daily
- **License**: Public Domain

### 3.2 AI4I 2020 Predictive Maintenance Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Description**: Synthetic dataset that reflects real predictive maintenance data encountered in industry
- **Labels**: Includes failure modes (tool wear failure, heat dissipation failure, power failure, overstrain failure)
- **Features**: 
  - Air temperature
  - Process temperature
  - Rotational speed
  - Torque
  - Tool wear
  - Different failure modes
- **Applications**: Building general predictive maintenance models for industrial applications, adaptable to automotive equipment

### 3.3 PHM Society Data Repository
- **Source**: [PHM Society](https://data.phmsociety.org/)
- **Description**: Collection of datasets for prognostics and health management, including some automotive applications
- **Labels**: Contains failure progressions and maintenance outcomes
- **Applications**: Developing algorithms for predictive maintenance and fault detection

## 4. Machine Learning Projects for Vehicle Diagnostics

### 4.1 ML-Based Vehicle Predictive Maintenance System
- **Source**: [GitHub - iDharshan/ML-Based-Vehicle-Predictive-Maintenance-System](https://github.com/iDharshan/ML-Based-Vehicle-Predictive-Maintenance-System-with-Real-Time-Visualization)
- **Description**: AI-driven predictive maintenance for vehicles using Gradient Boosting Machine (GBM) models on real-time sensor data
- **Features**: 
  - Real-time data visualization
  - Predicts maintenance probability
  - Estimates maintenance dates 2-3 weeks in advance
  - Provides probability percentages for potential part failures

### 4.2 Predictive Maintenance Metadata Repository
- **Source**: [GitHub - autonlab/pmx_data](https://github.com/autonlab/pmx_data)
- **Description**: Collection of datasets and scripts for predictive maintenance in various domains including automotive
- **Features**: Includes datasets for:
  - Time to failure prediction
  - Anomaly detection
  - Clustering
  - Fault detection and root cause analysis

### 4.3 BMW ML Predictive Maintenance Project
- **Source**: [GitHub - Explore-Dream-Discover/BMW_ML_Predictive_maintenance](https://github.com/Explore-Dream-Discover/BMW_ML_Predictive_maintenance)
- **Description**: Project focused specifically on BMW vehicle maintenance prediction

### 4.4 Mercedes-Benz Benchtime Prediction
- **Source**: [GitHub - GitKamo/Mercedes-Benz-benchtime-prediction](https://github.com/GitKamo/Mercedes-Benz-benchtime-prediction)
- **Description**: Project to optimize testing time for Mercedes-Benz vehicles using machine learning
- **Features**: Uses XGBoost and Random Forest regressors to predict testing time based on different permutations of features

### 4.5 Renault Telematics Security with Machine Learning
- **Source**: [Zenodo - Machine Learning Algorithm of Detection of DOS Attacks on an Automotive Telematic Unit](https://zenodo.org/records/2575338)
- **Description**: Algorithm developed by Renault Software Labs for detecting DoS attacks on automotive telematic units
- **Features**:
  - Uses unsupervised machine learning for attack detection
  - Low CPU requirements compatible with automotive hardware
  - Designed specifically for Renault's telematics systems

### 4.6 Koola.io Vehicle Predictive Maintenance
- **Source**: [Predicting Vehicle Maintenance using on-board and off-board data](https://ecosystem.evolve-h2020.eu/predicting-vehicle-maintenance-using-on-board-and-off-board-data/)
- **Description**: Research on combining DTC codes and OBD data for predictive maintenance
- **Use Cases**:
  - More accurate prediction of part failures (months in advance)
  - Optimized repair scheduling
  - Reduced vehicle downtime
  - Decreased fuel consumption and emissions
  - Supply chain optimization
  - Long-term customer retention
- **Machine Learning Approach**: Using historical repair data and DTCs to create predictive models

## 5. Related Automotive Sensor Datasets

### 5.1 RadarScenes Dataset
- **Source**: [RadarScenes Website](https://radar-scenes.com/)
- **Description**: Real-world radar point cloud dataset for automotive applications with data from four radar sensors
- **Features**: Over 4 hours of driving data with more than 7500 unique objects
- **Applications**: Object detection, tracking, segmentation

### 5.2 CARRADA Dataset
- **Source**: [CARRADA Dataset](https://arthurouaknine.github.io/codeanddata/carrada)
- **Description**: Synchronized camera and radar recordings with range-angle-Doppler annotations
- **Features**: Camera and automotive radar data with annotations
- **Applications**: Sensor fusion, object detection, autonomous driving

## 6. Creating Custom Labeled Datasets

For the CarSense ML service, you may need to create custom labeled datasets by:

### 6.1 Service Center Collaboration
- Partner with Romanian auto service centers to collect:
  - Pre-repair OBD2 data
  - DTC codes present
  - Actual repair actions taken
  - Post-repair validation data
  - Cost of repairs

### 6.2 Data Collection Strategy
- **Vehicle-Specific OBD2 Data**: Focus on collecting data from popular models in Romania (Dacia, BMW, VW, Renault)
- **DTC-Repair Correlation**: Gather historical data linking DTC codes to actual repair records
- **Environmental Factors**: Include Romania-specific environmental data that might affect vehicle performance
- **Dacia/Renault Focus**: Given the popularity of these brands in Romania, special attention should be given to collecting comprehensive diagnostic data from these vehicle models

### 6.3 Data Labeling Schema Example
```
{
  "vehicle_id": "unique_identifier",
  "make": "BMW",
  "model": "320d",
  "year": 2018,
  "engine_type": "diesel",
  "mileage_km": 85000,
  "dtc_codes": ["P0101", "P0234"],
  "obd_parameters": {
    "coolant_temp_c": 95,
    "engine_load_percent": 78,
    "intake_pressure_kpa": 110,
    ...
  },
  "repair_outcome": {
    "components_replaced": ["Mass Airflow Sensor", "Turbo pressure sensor"],
    "repair_success": true,
    "follow_up_required": false,
    "cost_eur": 350
  },
  "severity": "important",
  "downtime_days": 2
}
```

## 7. Model Development Priorities

1. **Fault Classification**: Build models to classify DTCs by severity and urgency
2. **Time-to-Failure Prediction**: Develop algorithms to predict when specific components might fail
3. **Maintenance Recommendation Engine**: Create a system that suggests optimal maintenance schedules
4. **Multi-brand Approach**: Develop separate models for different vehicle brands, with special focus on Dacia/Renault, BMW, and Mercedes models

## 8. Tools for OBD2 Data Collection

- **Hardware**: ELM327 adapters, KIWI 3 (PLX Devices), Carloop
- **Software**: OBD Auto Doctor, Torque Pro, CarScanner
- **DIY Solutions**: Raspberry Pi + OBD2 adapter, Arduino-based projects

## 9. Romanian Market Considerations

1. **Vehicle Fleet Analysis**: Understand the distribution of vehicle makes/models in Romania
2. **Regional Driving Patterns**: Account for specific driving conditions in Romanian cities and rural areas
3. **Seasonal Factors**: Consider how harsh winters or hot summers affect vehicle diagnostics
4. **Dacia Market Leadership**: Consider the high market share of Dacia vehicles in Romania when prioritizing development efforts

## 10. DTC-to-Repair Mapping Sources

### 10.1 Proprietary Sources (Requiring Licensing)
- **AllData** - Commercial repair database with comprehensive DTC-to-repair mappings
- **Mitchell1** - Professional automotive repair database with diagnostic flowcharts
- **Bosch ESI[tronic]** - Detailed European vehicle diagnostic information

### 10.2 Open Access Resources
- **Car Repair Forums** - Sites like BimmerForums (BMW), VWVortex, etc. contain user-reported DTC and repair outcomes
- **FixMyCar Community Data** - Some repair platforms share anonymized repair data
- **YouTube Diagnostic Channels** - Many mechanics document repair processes with initial DTCs

## 11. Additional Resources

- **NASA Prognostics Center of Excellence**: [nasa.gov/content/prognostics-center-of-excellence-data-set-repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **UCI Machine Learning Repository**: [archive.ics.uci.edu/ml/datasets.php](https://archive.ics.uci.edu/ml/datasets.php)
- **OpenML**: [openml.org](https://www.openml.org/)

## 12. Challenges and Considerations

- **Standardization**: Different vehicle manufacturers implement OBD2 differently
- **Completeness**: Not all PIDs are supported by all vehicles
- **Privacy**: Vehicle data may contain sensitive location information
- **Size**: High-frequency logging can generate large datasets, especially with multiple parameters
- **Annotation**: Labeling fault conditions can be challenging without ground truth
- **Integration Requirements**: 
  - Real-time processing capabilities for streaming vehicle data
  - Visualization tools for displaying diagnostic predictions
  - API services for mobile/web applications to access predictions
  - Brand-specific adapters for different manufacturer-specific diagnostic protocols

---

## License and Citation Requirements

If using academic datasets for research purposes, please cite the original authors. For example:

```
@article{roman2023evaluating,
  title={Evaluating the Privacy and Utility of Time-Series Data Perturbation Algorithms},
  author={Roman, Adrian-Silviu},
  journal={Mathematics},
  volume={11},
  number={5},
  pages={1260},
  year={2023},
  publisher={MDPI}
}
```

Some datasets have specific licensing restrictions. Always check the terms of use before incorporating into commercial applications. 